import cv2
import mediapipe as mp
import math
import time
import numpy as np
import threading
import winsound

# ---------------------------------------------------------
# ALARM SİSTEMİ (Multi-Threading)
# ---------------------------------------------------------
is_alarming = False

def play_alarm_sound():
    """Arka planda kamerayı dondurmadan BİİP sesi çalar."""
    global is_alarming
    is_alarming = True
    # 2500 Hz frekansta (yüksek, rahatsız edici), 500 milisaniye çal
    winsound.Beep(2500, 500)
    is_alarming = False

# ---------------------------------------------------------
# 1. MEDIAPIPE TASKS API KURULUMU
# ---------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)

RIGHT_EYE = [33, 160, 158, 133, 153, 144] 
LEFT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_ear(eye_points, landmarks, frame_w, frame_h):
    pts = []
    for index in eye_points:
        x = int(landmarks[index].x * frame_w)
        y = int(landmarks[index].y * frame_h)
        pts.append((x, y))

    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    h = euclidean_distance(pts[0], pts[3])

    if h == 0: return 0.0, pts
    ear = (v1 + v2) / (2.0 * h)
    return ear, pts

# ---------------------------------------------------------
# GERÇEK CNN MODELİ YÜKLEMESİ (Faz 2)
# ---------------------------------------------------------
from tensorflow.keras.models import load_model
model = load_model('drowsiness_model.h5')

def crop_eye_region(frame, eye_pts):
    """Göz bölgesini kırpıp CNN'in istediği 24x24 boyutuna getirir."""
    x_coords = [pt[0] for pt in eye_pts]
    y_coords = [pt[1] for pt in eye_pts]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    padding = 5
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, frame.shape[1])
    y_max = min(y_max + padding, frame.shape[0])
    
    eye_crop = frame[y_min:y_max, x_min:x_max]
    
    if eye_crop.size != 0:
        eye_crop = cv2.resize(eye_crop, (24, 24))
        eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    
    return eye_crop

def predict_eye_state(eye_img):
    """24x24 gri resmi CNN'e verip durumu tahmin eder.
    0: Closed_Eyes, 1: Open_Eyes"""
    if eye_img.size == 0: return 1 # Güvenlik için açık varsay
    
    # Keras'ın beklediği boyuta getir: (1, 24, 24, 1) ve normalize et
    img_array = eye_img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    return class_idx

def main():
    cap = cv2.VideoCapture(0)
    
    # ---------------------------------------------------------
    # ROLLING WINDOW (Yürüyen Pencere) MANTIĞI
    # ---------------------------------------------------------
    # Son 30 kareyi (yaklaşık 1 saniye) hafızada tutacağız
    # 1: Göz Kapalı, 0: Göz Açık
    frame_history = []
    MAX_FRAMES = 30 
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                right_ear, right_pts = calculate_ear(RIGHT_EYE, face_landmarks, frame_w, frame_h)
                left_ear, left_pts = calculate_ear(LEFT_EYE, face_landmarks, frame_w, frame_h)
                avg_ear = (right_ear + left_ear) / 2.0

                for pt in right_pts + left_pts:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)

                # ---------------------------------------------------------
                # AŞAMA 2 MANTIĞI: (EAR Kontrolü + GERÇEK CNN Tahmini)
                # ---------------------------------------------------------
                is_currently_closed = 0 # Varsayılan: Açık
                
                if avg_ear < 0.22:
                    # EAR tehlike sınırında. Kırpıp modele yolluyoruz.
                    right_eye_img = crop_eye_region(frame, right_pts)
                    left_eye_img = crop_eye_region(frame, left_pts)
                    
                    # Sağ ve sol gözü CNN'e sor: (0: Kapalı, 1: Açık)
                    r_pred = predict_eye_state(right_eye_img)
                    l_pred = predict_eye_state(left_eye_img)
                    
                    # Eğer iki gözden biri bile kapalı (0) çıkarsa, bu karede "Göz Kapalı" diyoruz.
                    if r_pred == 0 or l_pred == 0:
                        is_currently_closed = 1
                    
                    # Ekrana Debug Kırpılmış Gözü yansıtma
                    if right_eye_img.size != 0:
                        debug_eye = cv2.resize(right_eye_img, (100, 100))
                        frame[10:110, frame_w-110:frame_w-10] = cv2.cvtColor(debug_eye, cv2.COLOR_GRAY2BGR)

                # Buffer'a (Geçmişe) yeni kareyi ekle
                frame_history.append(is_currently_closed)
                if len(frame_history) > MAX_FRAMES:
                    frame_history.pop(0) # En eski kareyi sil
                
                # Risk Skoru: Son X karenin yüzde kaçı kapalıydı?
                closed_count = sum(frame_history)
                drowsy_score_percent = (closed_count / MAX_FRAMES) * 100.0

                # ---------------------------------------------------------
                # KULLANICI ARAYÜZÜ BİLDİRİMLERİ (UI) & SESLİ ALARM
                # ---------------------------------------------------------
                # Skor %40'ı geçerse (Blink değil, Micro-sleep başlangıcı)
                if drowsy_score_percent > 40:
                    cv2.putText(frame, "CNN: YORGUNLUK BELIRTISI!", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Skor %80'i geçerse (Alarm çal!)
                if drowsy_score_percent >= 80:
                    cv2.putText(frame, "UYARI: UYKU TESPITI!", (150, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.rectangle(frame, (0,0), (frame_w, frame_h), (0,0,255), 10)
                    
                    # Alarm şu an çalmıyorsa yeni bir Thread (İş parçacığı) başlat
                    global is_alarming
                    if not is_alarming:
                        t = threading.Thread(target=play_alarm_sound, daemon=True)
                        t.start()

                # Ana Gösterge Panelini Yazdır
                color = (0, 0, 255) if drowsy_score_percent > 50 else (255, 0, 0)
                cv2.putText(frame, f"EAR: {avg_ear:.2f} | Risk Skoru: %{int(drowsy_score_percent)}", 
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Asama 2: Multi-Stage Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
