import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Streamlit 페이지 설정
st.title('Face Mask Detection')
st.write('Upload an image to classify it as with_mask or without_mask.')

# 이미지 업로드 위젯
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    server_url = "http://localhost:8000/resnet/predict"  # FastAPI 서버 URL
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(server_url, files=files)

    # 결과 표시
    if response.status_code == 200:
        result = response.json()
        st.write(f'Prediction: {result["predicted_class"]}')
    else:
        st.write("Error in prediction")