import torch
from torchvision import transforms
from example.train_model.models.resnet34 import MasksModelFromResNet

class ModelHandler:
    def load_model(self):
        model = MasksModelFromResNet(len(self.class_names), pretrained=True, train_all_layers=True)
        #model = torch.load(f'{self.model_path}')
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'))['model_state_dict'])
        return model

    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

class DataHandler:
    def check_type(self, check_class, data):
        data = check_class(**data)
        
        return data
