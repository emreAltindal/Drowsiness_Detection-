# Drowsiness Detection Pipeline [OpenCV + MediaPipe + CNN]

Bu proje, sürücü uyku hali tespiti (Drowsiness Detection) için geliştirilmiş modern ve hibrit bir bilgisayarlı görü (Computer Vision) ve Derin Öğrenme (Deep Learning) boru hattıdır (pipeline).

Klasik Haar Cascades yöntemleri yerine, yüzdeki 478 noktayı çok daha düşük ışıkta bile yüksek performansla bulabilen modern **MediaPipe Tasks API (Face Landmarker)** ve özellik çıkarımı için kendi eğittiğimiz **Derin Öğrenme (CNN)** mimarisi harmanlanmıştır.

## Mimari Nasıl Çalışır? (Multi-Stage Pipeline)
Sistemin yavaşlamasını engellemek için çift aşamalı (filtreli) bir sistem kurulmuştur:

1. **Aşama 1 (Hız & Filtre):** 
   MediaPipe kameradan gelen görüntüyü analiz eder. Sağ ve sol gözün kilit noktalarını (landmarks) bularak Öklid mesafesi ile **EAR (Eye Aspect Ratio - Göz Açıklık Oranı)** değerini hesaplar. Bu işlem matematiksel olarak çok ucuz ve hızlıdır. 
   * EAR normal değerlerdeyse (Örn: > 0.22), sistem bir sonraki kareye (frame) atlar. CNN'i hiç yormaz.

2. **Aşama 2 (Kesinlik & CNN):**
   Eğer EAR tehlike sınırının altına düşerse, göz bölgesini çevreleyen pikseller bir makas gibi kırpılır (ROI Extraction), siyah-beyaz hale getirilir ve 24x24 boyutuna küçültülerek özel eğittiğimiz Keras (TensorFlow) CNN modeline (`drowsiness_model.h5`) gönderilir.
   * Model gözün "Açık (1)" veya "Kapalı (0)" olduğuna karar verir.

3. **Rolling Window (Yürüyen Pencere) & Multi-Threading Alarm:**
   Anlık göz kırpmaları (Blinking) ile gerçek uyuklamayı (Micro-sleep) ayırt etmek için sistem **Son 30 Kareyi (1 Saniye)** hafızasında tutar. 
   * Son 1 saniyenin %80'inden fazlasında göz kapalıysa sistem **Asenkron (Multi-Threading)** olarak Windows bip sesi (alarm) çalar. Threading kullanıldığı için alarm çalarken kamera görüntüsü asla donmaz.

## Kendi Modelinizi Eğitmek (Training)
Bu projede MRL Eye Dataset (Kaggle) içinden alınan 4000 (Açık/Kapalı) fotoğraf ile kendi CNN modelimiz eğitilmiştir. Eğitimi kendiniz yapmak isterseniz:
1. `prepare_dataset.py` dosyasını çalıştırarak verileri `dataset/` klasörüne hazırlayın.
2. `train.py` dosyasını çalıştırarak modeli 15 Epoch boyunca (ImageDataGenerator ile veri artırımı yapılarak) eğitin.
3. Çıkan `drowsiness_model.h5` dosyasını ana klasörde bırakın.

## Kurulum (Windows)

Bu proje sistemdeki diğer Python kütüphaneleriyle çakışmaması için izole bir **Sanal Ortam (Virtual Environment)** içinde çalışacak şekilde tasarlanmıştır.

1. Projeyi bilgisayarınıza indirin ve klasöre girin:
```bash
git clone <sizin-repo-linkiniz>
cd Drowsiness_Detection_Pipeline
```

2. Sanal ortamı oluşturun ve aktif edin:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Gerekli kütüphaneleri yükleyin (TensorFlow, OpenCV, MediaPipe vb. eklendi):
```bash
pip install -r requirements.txt
```

4. Projeyi çalıştırın:
```bash
python drowsiness_pipeline.py
```

## Geliştirici
- **Emre**
