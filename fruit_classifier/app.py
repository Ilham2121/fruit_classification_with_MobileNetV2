import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Inisialisasi model
model = tf.keras.models.load_model("C:/Users/Acer/fruitclassification/Model_saved/fruit_classifier_model.h5")

# Definisi label kelas
labels = ['apple', 'avocado', 'banana', 'cherry', 'kiwi',
        'mango', 'orange', 'stawberries', 'watermelon', 'pineapple'
]
# Inisialisasi state di luar fungsi main
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_prediction = None
    st.session_state.last_image = None
    st.session_state.is_reset = False

# Fungsi untuk memproses gambar yang diupload
def process_image(image):
    # Ubah gambar menjadi array numpy
    image_array = np.array(image)
    
    # Ubah ukuran gambar menjadi 224x224
    image_array = cv2.resize(image_array, (224, 224))
    
    # Normalisasi gambar
    image_array = image_array / 255.0
    
    # Tambahkan dimensi batch
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Fungsi untuk membuat prediksi
def make_prediction(image_array):
    # Buat prediksi menggunakan model
    predictions = model.predict(image_array)
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions[0])
    
    # Dapatkan nama kelas
    class_name = labels[predicted_class]
    
    return class_name

# Tambahkan tombol untuk mengupload gambar
st.title("Prediksi Gambar Buah")
st.write("Upload gambar buah untuk membuat prediksi")
image_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

# Jika gambar telah diupload
if image_file is not None:
    # Muat gambar
    image = Image.open(image_file)
    
    # Proses gambar
    image_array = process_image(image)
    
    # Buat prediksi
    class_name = make_prediction(image_array)
    
    # Tampilkan hasil prediksi
    st.write(f"Prediksi: {class_name}")

def main():
    st.title("Klasifikasi Buah Real-time")
    
    # Load model dengan compile=False
    try:
        model = tf.keras.models.load_model("C:/Users/Acer/fruitclassification/Model_saved/fruit_classifier_model.h5", compile=True)
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return
    
    # Definisi label kelas
    labels = [
        'apple', 'avocado', 'banana', 'cherry', 'kiwi',
        'mango', 'orange', 'stawberries', 'watermelon', 'pineapple'
    ]

    def detect_and_classify(frame):
        try:
            # Resize frame sesuai input model
            input_frame = cv2.resize(frame, (224, 224))
            input_frame = np.array(input_frame) / 255.0  # Normalisasi
            input_frame = np.expand_dims(input_frame, axis=0)

            # Prediksi menggunakan model dengan verbose=0
            predictions = model.predict(input_frame, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            return labels[predicted_class], confidence
            
        except Exception as e:
            st.error(f"Error dalam klasifikasi: {str(e)}")
            return None, 0.0

    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Tidak dapat mengakses kamera!")
        return

    # Placeholder untuk video feed
    frame_placeholder = st.empty()
    
    # Tambah kolom untuk tombol
    col1, col2 = st.columns(2)

    # Tombol reset
    if col2.button("Reset"):
        st.session_state.last_prediction = None
        st.session_state.last_image = None
        st.session_state.is_reset = True
        st.experimental_rerun()
    
    # Tombol klasifikasi
    if col1.button("Klasifikasi"):
        st.session_state.is_reset = False
        ret, frame = cap.read()
        if ret:
            # Dapatkan prediksi
            label, confidence = detect_and_classify(frame)
            
            if label is not None and confidence > 0.5:
                st.session_state.last_prediction = f"Buah terdeteksi: {label} (Confidence: {confidence:.2%})"
                st.session_state.last_image = frame.copy()
            else:
                st.session_state.last_prediction = "Tidak ada buah yang terdeteksi!"
                st.session_state.last_image = frame.copy()

    # Tampilkan hasil prediksi jika ada
    if not st.session_state.is_reset and st.session_state.last_image is not None:
        try:
            rgb_frame = cv2.cvtColor(st.session_state.last_image, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption=st.session_state.last_prediction or "Gambar terakhir")
        except Exception as e:
            st.error(f"Error saat menampilkan gambar: {str(e)}")

    # Video feed berkelanjutan
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame)
            else:
                break
    except Exception as e:
        st.error(f"Error pada video feed: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

if __name__ == "__main__":
    main()
