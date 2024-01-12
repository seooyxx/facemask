import argparse
import torch
from fastapi import FastAPI
from packages import resnet
from packages import FastAPIRunner
from packages.handler import ModelHandler

app = FastAPI()
model = None  # 전역 변수로 모델 선언

@app.on_event("startup")
async def load_model():
    global model
    handler = ModelHandler()
    model = handler.load_model()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

app.include_router(resnet)

@app.get('/')
def read_results():
    return {'msg' : 'Main'}
    
if __name__ == "__main__":
    # python main.py --host 127.0.0.1 --port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--port')
    args = parser.parse_args()
    api = FastAPIRunner(args)
    api.run()
    