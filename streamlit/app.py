import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title('Face Mask Detection')
st.write('Upload an image or capture one from your webcam to classify it as with_mask or without_mask.')

# 이미지 업로드 위젯
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# 웹캠 이미지 캡처 위젯
captured_image = st.camera_input("Or capture an image from your webcam")

# 이미지 처리 및 FastAPI 서버로 전송
def process_image(image_data):
    if image_data is not None:
        image = Image.open(image_data)
        st.image(image_data, caption='Processed Image', use_column_width=True)
        st.write("Classifying...")
        server_url = "http://localhost:8000/resnet/predict"
        files = {"file": image_data.getvalue()}
        response = requests.post(server_url, files=files)

    # 결과 표시
        if response.status_code == 200:
            result = response.json()
            st.write(f'Prediction: {result["predicted_class"]}')
        else:
            st.write("Error in prediction")

# 이미지 업로드 처리
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    process_image(uploaded_file)

# 웹캠 이미지 캡처 처리
if captured_image is not None:
     process_image(captured_image)
