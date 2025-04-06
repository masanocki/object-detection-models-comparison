import cv2
import tensorflow as tf
import numpy as np

# Załaduj model MobileNetV2
model = tf.keras.applications.MobileNetV2(
    weights="./pretrained_models/mobilenet-v2", input_shape=(224, 224, 3)
)


# Funkcja do przetwarzania każdej klatki wideo
def process_frame(frame):
    # Zmień rozmiar klatki do 224x224, ponieważ to rozmiar wejścia modelu MobileNetV2
    resized_frame = cv2.resize(frame, (224, 224))

    # Przekształć klatkę na tablicę NumPy
    img_array = np.array(resized_frame)

    # Przekształć obraz do postaci akceptowanej przez MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Dodaj wymiar wsadu (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Wykonaj predykcję
    predictions = model.predict(img_array)

    # Odbierz wynik klasyfikacji
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions
    )

    # Zwróć najlepsze rozpoznane obiekty (możesz dodać inne operacje, jak rysowanie na obrazie)
    return (
        decoded_predictions[0][0][1],
        decoded_predictions[0][0][2],
    )  # Nazwa i pewność predykcji


# Ścieżka do wideo
video_path = "./dataset/rugby/video/1.avi"

# Otwórz wideo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Przetwórz klatkę
    label, confidence = process_frame(frame)

    # Dodaj napis z wynikami na klatce wideo
    cv2.putText(
        frame,
        f"{label}: {confidence:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Wyświetl klatkę
    cv2.imshow("Video Detection", frame)

    # Jeśli naciśniesz 'q', przerwij
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()
