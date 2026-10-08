import os
import zipfile
import random

ZIP_FILE_PATH = r"C:\Users\emrea\Downloads\archive_8.zip"
TARGET_OPEN_DIR = r"dataset\Open_Eyes"
TARGET_CLOSED_DIR = r"dataset\Closed_Eyes"
NUM_IMAGES_TO_COPY = 2000

def prepare_data_from_zip():
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"HATA: ZIP dosyasi bulunamadi: {ZIP_FILE_PATH}")
        return

    os.makedirs(TARGET_OPEN_DIR, exist_ok=True)
    os.makedirs(TARGET_CLOSED_DIR, exist_ok=True)

    print(f"ZIP dosyasi okunuyor: {ZIP_FILE_PATH}")
    print("Bu islem birkac saniye surebilir...\n")

    with zipfile.ZipFile(ZIP_FILE_PATH, "r") as z:
        all_files = z.namelist()
        image_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        
        # Dosya yollarindan awake ve sleepy kelimelerini ara
        open_candidates = [f for f in image_files if "awake" in f.lower()]
        closed_candidates = [f for f in image_files if "sleepy" in f.lower()]

        print(f"ZIP İcinde Bulunan Toplam Acik Goz (awake): {len(open_candidates)}")
        print(f"ZIP İcinde Bulunan Toplam Kapali Goz (sleepy): {len(closed_candidates)}\n")

        # Rasgele secim
        selected_open = random.sample(open_candidates, min(NUM_IMAGES_TO_COPY, len(open_candidates)))
        selected_closed = random.sample(closed_candidates, min(NUM_IMAGES_TO_COPY, len(closed_candidates)))

        print(f"{len(selected_open)} Acik Goz cikartiliyor -> {TARGET_OPEN_DIR}")
        for file_path in selected_open:
            file_data = z.read(file_path)
            base_name = os.path.basename(file_path)
            with open(os.path.join(TARGET_OPEN_DIR, base_name), "wb") as f:
                f.write(file_data)

        print(f"{len(selected_closed)} Kapali Goz cikartiliyor -> {TARGET_CLOSED_DIR}")
        for file_path in selected_closed:
            file_data = z.read(file_path)
            base_name = os.path.basename(file_path)
            with open(os.path.join(TARGET_CLOSED_DIR, base_name), "wb") as f:
                f.write(file_data)

    print("\nBitti! Resimler hazir.")

if __name__ == "__main__":
    prepare_data_from_zip()

