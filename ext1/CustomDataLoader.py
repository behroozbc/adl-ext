from torch.utils.data import Dataset
import random
from torchvision import datasets, transforms
class CustomDataLoader(Dataset):
    def __init__(self,dataset,crop_prob=0.25) -> None:
        super().__init__()
        self.dataset=dataset
        self.crop_prob=crop_prob
        self.transform = transforms.RandomCrop(size=(224, 224))
        self.transformCompose=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ) ])
    def __len__(self): return len(self.dataset)
    
    def __getitem__(self, idx): 
        image, label = self.dataset[idx] 
        if random.random() < self.crop_prob:
            image = self.transform(image)
            image = self.transformCompose(image)
            return image, label