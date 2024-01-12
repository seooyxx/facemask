import os

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from pydantic import Field
from pydantic import validator
from packages.handler import ModelHandler


class ProjectConfig(ModelHandler):
    def __init__(self, model_type='epoch2'):
        self.model_type = model_type
        self.project_path = os.path.abspath(os.getcwd())
        self.add_example_path = "example/train_model"
        self.model_path = f"{self.project_path}/models/epoch2.pth"
        self.class_names = ['with_mask', 'without_mask']
        ModelHandler.__init__(self)


class VariableConfig:
    def __init__(self):
        self.host_list = ['127.0.0.1', '0.0.0.0']
        self.port_list = ['8000', '8088']


class APIEnvConfig(BaseSettings):
    host: str = Field(default='0.0.0.0', env='api host')
    port: int = Field(default='8000', env='api server port')
    
    @validator("host", pre=True)
    def check_host(cls, host_input):
        if host_input == 'localhost':
            host_input = "127.0.0.1"
        if host_input not in VariableConfig().host_list:
            raise ValueError("host error")
        return host_input
    
    @validator("port", pre=True)
    def check_port(cls, port_input):
        if port_input not in VariableConfig().port_list:
            raise ValueError("port error")
        return port_input


class APIConfig(BaseModel):
    api_name: str = 'main:app'
    api_info: APIEnvConfig = APIEnvConfig()
