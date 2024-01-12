import argparse
import os

from fastapi import FastAPI
from packages import resnet
from packages import FastAPIRunner

app = FastAPI()

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
    