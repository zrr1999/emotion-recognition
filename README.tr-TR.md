# Çok Modlu Duygu Tanıma (Multimodal Emotion Recognition)
中文 | [English](README.en-US.md)

> Derin öğrenme tabanlı çok modlu duygu tanıma yöntemleri araştırma paketi, metin, ses ve video üçlü modda duygu sınıflandırma ve duygusal analiz görevlerini destekler.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Proje Tanıtımı

Bu proje, çok modlu duygu tanıma için bir araştırma paketidir ve aşağıdaki işlevleri destekler:

- 🎯 **Çok modlu entegrasyon**: Metin(T), Ses(A), Video(V) üçlü modun ayrı ve birlikte kullanımını destekler
- 📊 **Çoklu veri seti desteği**: MELD, IEMOCAP, SIMS gibi popüler duygu veri setlerini destekler
- 🚀 **Verimli eğitim**: Önbellek mekanizması, karışık hassasiyetli eğitim, bilgi distilasyonu gibi optimizasyon tekniklerini entegre eder
- ⚙️ **Esnek yapılandırma**: TOML yapılandırma dosyalarına dayalı modüler tasarım, çeşitli deney yapılandırmalarını destekler
- 📈 **Tam işlem akışı**: Veri ön işleme'den model eğitimine, değerlendirmeye kadar tam araç zincir

Proje yapısı hakkında daha fazla bilgi için [proje yapısı belgesine](docs/structure.md) bakın.


## Hızlı Başlangıç

### Ortam Gereksinimleri

- Python 3.12+
- PyTorch 2.4.1
- CUDA 12.1 (isteğe bağlı, GPU hızlandırma için)

### Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/zrr1999/emotion-recognition.git
cd emotion-recognition
```

2. Bağımlılıkları yükleyin:
```bash
# uv ile yükleme (önerilir)
uv sync --all-extras --dev

# Yerel kullanıcılar Tsinghua kaynağını kullanarak hızlandırabilir
uv sync --all-extras --dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
```

3. Veri setinizi hazırlayın:
```bash
# Veri seti dizinini oluşturun
mkdir datasets

# MELD veri setini indirin ve bağlayın
ln -s /path/to/MELD datasets/MELD
```

Veri seti [proje yapısı belgesinde](docs/structure.md#veri-seti-yapısı) belirtilen formatta organize edilmelidir.

### Temel Kullanım

#### Komut Satırı Araçları

Proje iki adet komut satırı aracı sağlar:

```bash
# Duygu tanıma aracı
emotion-recognize --help

# Veri işleme aracı
emotion-tool --help
```

#### Model Eğitimi

[nanoflow](https://github.com/zrr-lab/nanoflow) kullanarak deneylerinizi çalıştırın:

```bash
# Temel deneyleri çalıştırın
uvx nanoflow run experiments/mdn.toml
uvx nanoflow run experiments/makd.toml

# Ayrımcı deneyleri çalıştırın
uvx nanoflow run experiments/mdn-ablation.toml
uvx nanoflow run experiments/makd-ablation.toml
```

## Temel Özellikler

### 🚀 Verimli Eğitim Mekanizması

#### Özellik Önbellekleme Mekanizması
Özellik çıkarma modülü dondurulduğunda, aynı girdiye karşılık gelen özellik çıkarma modülünün çıktısı değişmez, bu nedenle özellik çıkarma modülünün çıktısını önbelleğe alabiliriz, böylece tekrarlayan hesaplamalardan kaçınılabilir. Bu proje [SafeTensors](https://github.com/huggingface/safetensors) formatını kullanarak özellik önbelleklemesi yapar, diğer formatlara göre daha iyi performansa ve güvenliğe sahiptir.

#### Şeffaf Model Depolama
Deneyler sırasında, genellikle birçok modelin kontrol noktasını kaydederiz, ayrıca farklı modeller eğitmek için parametreleri değiştirin. Genellikle her bir model için ayrı ayrı parametreler kaydetmek depolama alanının israfına yol açar. Bu sorunu çözmek için, aynı model parametrelerini farklı model klasörlerine yumuşak bağlantılarla bağlarız, böylece depolama alanındaki israf azaltılır. Ayrıca bu yöntem orijinal klasör yapısını bozmadan kalır, böylece genel yapı daha net hale gelir.

#### Karışık Hassasiyetli Eğitim
Otomatik karışık hassasiyet (AMP) desteği entegre edilmiştir, model hassasiyetini korurken eğitim sürecini önemli ölçüde hızlandırır ve bellek kullanımını azaltır.

### ⚙️ Esnek Yapılandırma Sistemi

#### Dinamik Yapılandırma Fikri
Deneyler sırasında, farklı modelleri karşılaştırmak için genellikle parametreleri değiştirerek birden fazla farklı model eğitmek gerekir. Bu proje yapılandırma dosyası çözümünü benimser, kodu temiz tutarken parametreleri kolayca değiştirebilirsiniz. Yapılandırma dosyaları liste, sözlük gibi karmaşık veri yapılarını destekler, ifade gücü komut satırı parametrelerinden çok daha yüksektir.

TOML yapılandırma dosyaları kullanarak modüler tasarım uygulanmıştır:

- `configs/encoders/`: Kodlayıcı yapılandırmaları (T, A, V ve kombinasyonları)
- `configs/fusion/`: Entegrasyon stratejileri yapılandırmaları
- `configs/losses/`: Kayıp fonksiyonları yapılandırmaları
- `configs/dataset/`: Veri seti yapılandırmaları

Teknoloji bağımlılığı: [Pydantic](https://pydantic-docs.helpmanual.io/) yapılandırma doğrulama ve tür kontrolü için.

### 🔄 Bilgi Distilasyonu Optimizasyonu

[TelME](https://github.com/yuntaeyang/TelME) uygulamasında, farklı modelleri bilgi distilasyonuyla eğitmek birkaç kez gerektirir, bu da eğitim süresini çok uzun hale getirir (bu proje önbellekleme teknolojisini kullansa bile). Bu sorunu çözmek için, birden fazla modun bilgi distilasyonu eğitimini tek bir eğitimde birleştiririz, böylece eğitim süresi önemli ölçüde azaltılabilir.

### 📊 Çoklu Veri Seti Desteği

Popüler duygu tanıma veri setlerini destekler:
- **MELD**: Duygu sınıflandırma ve duygusal analiz
- **IEMOCAP**: Duygu sınıflandırma
- **SIMS**: Duygusal analiz

### 🔧 Sürekli Entegrasyon

Proje temel düzeyde sürekli entegrasyonu uygular, detaylar için [train-and-eval.yml](.github/workflows/train-and-eval.yml) dosyasına bakabilirsiniz.

## Proje Kuralları

### Kontrol Noktası Adlandırma Kuralları

Kontrol noktası adlandırma formatı `{eğitim türü}/{veri seti}/{eğitim yöntemi}--{grup boyutu}--{sınıflandırma kayıp fonksiyonu}/{ağ özet}/{ağ hash}--{rastgele tohum}` şeklindedir, örneğin `training/MELD--E/trainable--2--{loss}/1xE--T/51fe7ba3--114`.

#### Mod Kısaltmaları
- **T** (Text): Metin mod
- **A** (Audio): Ses mod
- **V** (Video): Video mod

#### Eğitim Yöntemi Kısaltmaları
- **T** (Full Tuning): Tam parametre ince ayar
- **L** (LoRA): Düşük sıralı uyum
- **F** (Froze Backbones): Sabit sütun

#### Veri Seti Türü Kısaltmaları
- **E** (Emotion): Duygu sınıflandırma görevi
- **S** (Sentiment): Duygusal analiz görevi

## Teknik Mimari

### Temel Bağımlılıklar

| Bileşen | Versiyon | Kullanım |
|------|------|------|
| Python | 3.12+ | Çalıştırma ortamı |
| PyTorch | 2.4.1 | Derin öğrenme çerçevesi |
| Transformers | 4.48+ | Önceden eğitilmiş model desteği |
| Pydantic | 2.0+ | Yapılandırma doğrulama |
| SafeTensors | 0.4+ | Verimli özellik önbellekleme |

### Model Mimarisi

```
Çok modlu girdi → Özellik kodlayıcı → Özellik entegrasyonu → Sınıflandırıcı → Duygu tahmini
    ↓           ↓          ↓        ↓         ↓
  T/A/V    BERT/Whisper  Deep/MoE  Linear   Sınıflar
```

#### Kodlayıcı Desteği
- **Metin(T)**: BERT, RoBERTa, ModernBERT vb.
- **Ses(A)**: Whisper, Distil-Whisper vb.
- **Video(V)**: OpenCV özellik çıkarma

#### Entegrasyon Stratejileri
- **Vanilla entegrasyon**: Basit birleştirme
- **Derin entegrasyon**: Derin sinir ağları
- **MoE entegrasyon**: Karmaşık uzman modeli
- **Dikkat entegrasyonu**: Kendi dikkat mekanizması

### Performans Optimizasyonu

- **Video okuma**: OpenCV (0.08s/çerçeve) vs PyAV (0.17s/çerçeve)
- **Özellik önbellekleme**: SafeTensors formatı kullanımı, sabitken tekrarlayan hesaplamalardan kaçınma
- **Depolama optimizasyonu**: Yumuşak bağlantı mekanizması ile model depolama alanı azaltma

## Deney Sonuçları

> Aşağıdaki sonuçlar MELD veri setindeki duygu sınıflandırma görevine dayanmaktadır

### Tek Mod Performansı

#### Metin Modu (Metin yalnızca)

| Yöntem | Rastgele tohum | Doğruluk | Weighted-F1 |
|------|------|--------|-------------|
| APCL (temp=0.08, β=0.1, γ=0.1) | 43 | 67.74% | 67.04% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 42 | 68.05% | 66.91% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 114 | 67.59% | 66.55% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 0 | 67.93% | 66.92% |
| SPCL (temp=0.08, pool=512, support=64) | 42 | 68.31% | 67.31% |
| SPCL (temp=0.08, pool=512, support=64) | 114 | 67.32% | 66.55% |
| SPCL (temp=0.08, pool=512, support=64) | 0 | 66.63% | 66.50% |

### Çok Mod Performansı

| Mod Kombinasyonu | Yöntem | Doğruluk | Weighted-F1 | Notlar |
|----------|------|--------|-------------|------|
| T+A+V | doldurulacak | - | - | doldurulacak |
| T+A | doldurulacak | - | - | doldurulacak |
| T+V | doldurulacak | - | - | doldurulacak |


## Katkı Kılavuzu

Kod ve fikirlerinizi memnuiyetle kabul ederiz! Aşağıdaki adımları izleyin:

1. Bu depoyu fork edin
2. Özellik dalını oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişiklikleri gönderin (`git commit -m 'Add amazing feature'`)
4. Dalı itin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

### Geliştirme Ortamı

```bash
# Geliştirme bağımlılıklarını yükleyin
uv sync --dev

# Kod kontrolünü çalıştırın
uv run ruff check src/
uv run pyright src/

# Biçimlendirmeyi çalıştırın
uv run ruff format src/
```

## Referanslar

### Önceden Eğitilmiş Modeller
- [BERT: Derin Çift Yönlü Dönüştürücülerin Önceden Eğitimi](https://arxiv.org/pdf/1810.04805)
- [RoBERTa: Güçlü Bir BERT Önceden Eğitim Yaklaşımı](https://arxiv.org/pdf/1907.11692)
- [ModernBERT: BERT Mimarisini Modernleştirme](https://github.com/AnswerDotAI/ModernBERT)

### Model Sıkıştırma Teknikleri
- [LoRA: Büyük Dil Modellerinin Düşük Sıralı Uyumu](https://huggingface.co/docs/peft/task_guides/lora_based_methods)

### Bilgi Distilasyonu
- [Derin Öğrenme için Bilgi Distilasyonu](https://arxiv.org/pdf/2104.09044)
- [DIST: Öğrenci-Oğretmen Ağları ile Bilgi Distilasyonu](https://arxiv.org/pdf/2205.10536)
- [Çapraz Modlu Bilgi Distilasyonu](https://arxiv.org/pdf/2401.12987v2)

### Karmaşık Uzman Modelleri
- [TGMoE: Metin Yönlendirilmiş Bir Karmaşık Uzman Modeli](https://ftp.saiconference.com/Downloads/Volume15No8/Paper_119-TGMoE_A_Text_Guided_Mixture_of_Experts_Model.pdf)

### Ses Tanıma
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Distil-Whisper: Güçlü Bilgi Distilasyonu](https://arxiv.org/abs/2311.00430)
- [Distil-Large-V3](https://huggingface.co/distil-whisper/distil-large-v3)
- [BELLE: Herkesin Büyük Dil Modeli Motoru](https://github.com/LianjiaTech/BELLE)

## Diğer Referanslar

### Teknik Blog ve Dersler
- [Çok Modlu Duygu Tanıma Üzerine Genel Bakış](https://zhuanlan.zhihu.com/p/694747931) - Çok modlu duygu tanıma teknolojisinin kapsamlı tanıtımı
- [Otomatik Karışık Hassasiyetli AMP Eğitimi Açıklaması](https://zhuanlan.zhihu.com/p/408610877/) - PyTorch AMP teknolojisi açıklaması ve en iyi uygulamalar

### Veri Setleri ve Referanslar
- [Duygu Tanıma Veri Seti Karşılaştırması](https://paperswithcode.com/task/emotion-recognition-in-conversation) - Papers with Code üzerindeki ilgili kaynaklar
- [MELD Veri Seti Resmi Sitesi](https://affective-meld.github.io/) - MELD veri seti resmi sitesi
- [IEMOCAP Veri Seti](https://sail.usc.edu/iemocap/) - IEMOCAP duygu veri tabanı

### Akademik Kaynaklar
- [Çok Modlu Öğrenme Yaklaşımları Üzerine Genel Bakış](https://arxiv.org/abs/2209.05025) - Çok modlu derin öğrenmenin en yeni gelişmeleri
- [Duygusal Hesaplama Konferansı (ACII)](http://acii-conf.org/) - Duygusal hesaplama alanının önde gelen konferansı
- [Çok Modlu Makine Öğrenimi Dersleri](https://cmu-multicomp-lab.github.io/mmml-tutorial/) - CMU Çok Modlu Dersleri

### Araçlar ve Çerçeveler
- [Transformers Kütüphanesi Belgeleri](https://huggingface.co/docs/transformers/) - HuggingFace Transformers kullanım kılavuzu
- [PyTorch Resmi Dersleri](https://pytorch.org/tutorials/) - PyTorch Derin Öğrenme Dersleri
- [Nanoflow İş Akışı Motoru](https://github.com/zrr-lab/nanoflow) - Deney yönetimi aracı

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## Teşekkürler

Bu projeye yönlendirme ve fikirlerini sunan tüm arkadaşlarımıza ve öğretmenlerimize teşekkür ederiz!

---

Sorunlarınız veya önerileriniz varsa, [Issue](https://github.com/zrr1999/emotion-recognition/issues) veya [Pull Request](https://github.com/zrr1999/emotion-recognition/pulls) gönderin.
