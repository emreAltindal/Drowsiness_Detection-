import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_drowsiness_cnn():
    """
    Sürücü uyku hali tespiti için kullanılacak hafif (lightweight) CNN modeli.
    Girdi: 24x24 piksel, tek kanallı (siyah-beyaz/grayscale) göz fotoğrafı.
    Çıktı: 2 sınıf (0: Kapalı, 1: Açık)
    """
    model = Sequential()

    # 1. Evrişim (Convolution) Katmanı: 
    # Resimdeki temel çizgileri, kenarları (edge detection) öğrenir.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # 2. Evrişim Katmanı:
    # Daha karmaşık dokuları (göz kapağı kıvrımları, kirpikler) öğrenir.
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # 3. Evrişim Katmanı:
    # Gözün kapalı mı açık mı olduğunu belirleyen derin özellikleri çıkarır.
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Vektöre Dönüştürme (Flatten)
    # Matris halindeki pikselleri tek boyutlu uzun bir listeye çeviririz.
    model.add(Flatten())

    # Yapay Sinir Ağı (Dense) Katmanı: 
    # Çıkarılan özellikleri kullanarak karar mekanizmasını kurar.
    model.add(Dense(128, activation='relu'))
    
    # Dropout: Ezberlemeyi (Overfitting) önlemek için nöronların %50'sini rastgele kapatır.
    model.add(Dropout(0.5))

    # Çıkış Katmanı: 
    # 2 nöron (Kapalı ve Açık). Softmax, oran verir (Örn: %90 Kapalı, %10 Açık)
    model.add(Dense(2, activation='softmax'))

    # Modeli Derleme (Compile)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_drowsiness_cnn()
    model.summary()  # Modelin yapısını ve parametre sayısını ekrana yazdırır
