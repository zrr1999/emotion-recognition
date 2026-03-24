from __future__ import annotations

import os
import re
import shlex
import tomllib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_PROJECT_ROOT = Path("/root/emotion-recognition")
REMOTE_HF_CACHE_ROOT = Path("/vol/hf")
REMOTE_DATASETS_ROOT = REMOTE_PROJECT_ROOT / "datasets"
REMOTE_CHECKPOINTS_ROOT = REMOTE_PROJECT_ROOT / "checkpoints"

HF_CACHE_VOLUME_NAME = "emotion-recognition-hf-cache"
DATASETS_VOLUME_NAME = "emotion-recognition-datasets"
CHECKPOINTS_VOLUME_NAME = "emotion-recognition-checkpoints"

DEFAULT_MODEL_IDS = [
    "roberta-large",
    "facebook/data2vec-audio-base-960h",
    "MCG-NJU/videomae-base-finetuned-kinetics",
]
DEFAULT_SMOKE_CONFIG_PATHS = [
    "configs/dataset/MELD--E.toml",
    "configs/encoders/T+A+V.toml",
    "configs/fusion/DF-1.0.toml",
    "configs/fusion/kwargs/attn.toml",
    "configs/losses/classification/weight.toml",
]


def _dependency_name(specifier: str) -> str:
    return re.split(r"[<>=!~;\\[ ]", specifier, maxsplit=1)[0]


def _load_modal_dependencies() -> tuple[list[str], str]:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    project = pyproject["project"]
    dependencies = [
        *project["dependencies"],
        *project["optional-dependencies"]["train"],
        *project["optional-dependencies"]["modal"],
    ]

    filtered_dependencies: list[str] = []
    torch_dependency: str | None = None
    seen_dependencies: set[str] = set()
    for dependency in dependencies:
        dependency_name = _dependency_name(dependency)
        if dependency_name == "emotion-recognition-utils":
            continue
        if dependency_name == "torch":
            torch_dependency = dependency
            continue
        if dependency_name in seen_dependencies:
            continue
        seen_dependencies.add(dependency_name)
        filtered_dependencies.append(dependency)

    if torch_dependency is None:
        raise ValueError("torch dependency not found in pyproject.toml")
    return filtered_dependencies, torch_dependency


def _recursive_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_training_config_dict(config_paths: Sequence[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for config_path in config_paths:
        with open(PROJECT_ROOT / config_path, "rb") as file:
            _recursive_update(config, tomllib.load(file))
    return config


def _resolve_model_ids(config_paths: Sequence[str] | None, model_ids: Sequence[str] | None) -> list[str]:
    resolved_model_ids = list(DEFAULT_MODEL_IDS)
    selected_config_paths = list(config_paths or DEFAULT_SMOKE_CONFIG_PATHS)
    if selected_config_paths:
        config = _load_training_config_dict(selected_config_paths)
        encoder_configs = config.get("model", {}).get("encoder", {})
        for encoder_config in encoder_configs.values():
            if isinstance(encoder_config, dict) and isinstance(encoder_config.get("model"), str):
                resolved_model_ids.append(encoder_config["model"])
    if model_ids is not None:
        resolved_model_ids.extend(model_ids)
    return list(dict.fromkeys(resolved_model_ids))


REMOTE_ENV = {
    "HF_HOME": REMOTE_HF_CACHE_ROOT.as_posix(),
    "HF_HUB_CACHE": (REMOTE_HF_CACHE_ROOT / "hub").as_posix(),
    "PYTHONPATH": ":".join(
        [
            (REMOTE_PROJECT_ROOT / "src").as_posix(),
            (REMOTE_PROJECT_ROOT / "packages" / "emotion-recognition-utils" / "src").as_posix(),
        ]
    ),
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": (REMOTE_HF_CACHE_ROOT / "hub").as_posix(),
}

IMAGE_DEPENDENCIES, TORCH_DEPENDENCY = _load_modal_dependencies()

hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
datasets_volume = modal.Volume.from_name(DATASETS_VOLUME_NAME, create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(CHECKPOINTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libsndfile1")
    .uv_pip_install(*IMAGE_DEPENDENCIES)
    .run_commands(
        "python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 "
        f"{shlex.quote(TORCH_DEPENDENCY)}"
    )
    .add_local_dir(PROJECT_ROOT.as_posix(), remote_path=REMOTE_PROJECT_ROOT.as_posix())
    .env(REMOTE_ENV)
    .workdir(REMOTE_PROJECT_ROOT.as_posix())
)

app = modal.App(name="emotion-recognition")

BASE_VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    REMOTE_HF_CACHE_ROOT.as_posix(): hf_cache_volume,
    REMOTE_DATASETS_ROOT.as_posix(): datasets_volume,
    REMOTE_CHECKPOINTS_ROOT.as_posix(): checkpoints_volume,
}


@app.function(image=image, cpu=2.0, timeout=1800, volumes=BASE_VOLUMES)
def warm_hf_cache(
    config_paths: list[str] | None = None,
    model_ids: list[str] | None = None,
) -> list[str]:
    from huggingface_hub import snapshot_download

    resolved_model_ids = _resolve_model_ids(config_paths, model_ids)
    cache_dir = REMOTE_HF_CACHE_ROOT / "hub"
    for model_id in resolved_model_ids:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx"],
        )

    hf_cache_volume.commit()
    return resolved_model_ids


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes=BASE_VOLUMES,
)
def train_smoke(
    config_paths: list[str] | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    num_epochs: int = 1,
    max_train_batches: int = 10,
    eval_interval: int = 1,
    checkpoint: str | None = None,
) -> str:
    from recognize_cli.cli_recognize import train

    os.chdir(REMOTE_PROJECT_ROOT)
    resolved_config_paths = [Path(path) for path in config_paths or DEFAULT_SMOKE_CONFIG_PATHS]
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None

    train(
        resolved_config_paths,
        batch_size=batch_size,
        seed=seed,
        checkpoint=checkpoint_path,
        num_epochs=num_epochs,
        max_train_batches=max_train_batches,
        eval_interval=eval_interval,
    )

    hf_cache_volume.commit()
    checkpoints_volume.commit()
    if checkpoint_path is not None:
        return checkpoint_path.as_posix()
    return REMOTE_CHECKPOINTS_ROOT.as_posix()
