import os
import torch
import torchvision

def build_resnet(input_dimensions = [1, 280, 360], file_path = '', rank=0):

    resnet18 = torchvision.models.resnet18(pretrained=False)
    resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    resnet18.fc = torch.nn.Linear(512, 2, bias=True)
    
    if file_path != '':
        state_dict = torch.load(file_path, map_location=lambda s, loc: s)
        resnet18.load_state_dict(state_dict, strict=True)
        
    resnet18 = resnet18.to(rank)
    
    return resnet18