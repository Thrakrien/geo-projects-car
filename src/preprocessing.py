import os
from torchvision import transforms
# from dotenv import load_dotenv
from src.dataset import PatchifySegmentationDataset, PatchifyInference

class PreProcessingImage:
    def __init__(self, transform_type):
        self.transform_type = transform_type

    def transformations(self):
        if self.transform_type == "train":
            transformed_data = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            return transformed_data

        elif self.transform_type == "validation":
            transformed_data = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
            return transformed_data
        
        else:
            raise ValueError("Tipo de Transformação Inválido!!!")
