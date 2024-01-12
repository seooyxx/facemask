from fastapi import APIRouter, File, UploadFile
import torch
from PIL import Image
from io import BytesIO
from main import model

# from packages.config import DataInput, PredictOutput
from packages.config import ProjectConfig


# Project config 설정
project_config = ProjectConfig('resnet')
# 모델 가져오기
#model = project_config.load_model()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)
#model.eval()

resnet = APIRouter(prefix='/resnet')

# router 마다 경로 설정
@resnet.get('/', tags=['resnet'])
async def start_resnet():
    return {'msg' : 'Here is resnet'}

@resnet.post('/predict', tags=['resnet'])
async def resnet_predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    transform = project_config.get_transform()
    transformed_image = transform(image).unsqueeze(0).to(device)
    # Inference 수행
    with torch.no_grad():
        output = model(transformed_image)
        prediction = int(output >= 0.5)

    # 결과 반환
    return {'predicted_class': project_config.class_names[prediction]}
