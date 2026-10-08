import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import build_drowsiness_cnn

# ==============================================================================
# AYARLAR
# ==============================================================================
DATASET_DIR = "dataset"
BATCH_SIZE = 32
TARGET_SIZE = (24, 24)
EPOCHS = 15

def train_model():
    print("--- Yapay Zeka (CNN) Eğitimi Başlıyor ---")
    
    # 1. Veri Artırma (Data Augmentation) ve Okuma
    # Gerçek hayatta kamera bazen yamuk durabilir, ışık farklı olabilir.
    # ImageDataGenerator resimleri hafifçe çevirerek, yakınlaştırarak yapay zekanın ezberlemesini önler (daha zeki yapar).
    datagen = ImageDataGenerator(
        rescale=1./255,          # Pikselleri 0-255 arasından 0-1 arasına çeker (Daha hızlı öğrenir)
        rotation_range=15,       # Resmi %15 döndür
        zoom_range=0.1,          # %10 yakınlaştır
        validation_split=0.2     # Verilerin %20'sini test/onay için ayır
    )

    print("\nEğitim (Training) verileri yükleniyor...")
    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=TARGET_SIZE,
        color_mode="grayscale",  # Modelimiz Siyah-Beyaz (24x24) tasarlandı
        batch_size=BATCH_SIZE,
        class_mode="categorical",# Kapalı ve Açık olmak üzere 2 kategori
        subset="training"
    )

    print("\nTest (Validation) verileri yükleniyor...")
    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=TARGET_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    # 2. cnn_model.py içindeki mimarimizi çağırıyoruz
    model = build_drowsiness_cnn()
    
    print("\nEğitim Başlıyor! (Lütfen bitene kadar bekleyin...)")
    
    # 3. Model Eğitimi (Training)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # 4. Eğitilen Modeli Kaydetme (Beyin Dosyası)
    model.save("drowsiness_model.h5")
    print("\nBAŞARILI! Model eğitildi ve 'drowsiness_model.h5' adıyla kaydedildi.")
    
    # Hangi sınıfın 0, hangisinin 1 olduğunu görelim (Pipeline entegrasyonu için önemli)
    print("\nSınıf Etiketleri:", train_generator.class_indices)

if __name__ == "__main__":
    train_model()
