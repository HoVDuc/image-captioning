import os
import torch
import torchvision
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import Dataset
from PIL import Image

class Flickr8kDataset(Dataset):
    def __init__(self, path, data, transform, device):
        self.path = path
        self.data = data
        self.transform = transform
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        filename = self.data['image'][index]
        caption = self.data['caption'][index]
        image_path = os.path.join('Images', filename)
        image = Image.open(os.path.join(self.path, image_path))
        transform = torchvision.transforms.Compose([
            Resize(torch.Size([224, 224])),
            ToTensor()
        ])
        image = transform(image)
        return image, caption
        
        