import os
# from dotenv import load_dotenv
from src.preprocessing import PreProcessingImage

img_sat = "/data/integracar/amostras_car_orotofoto/"
img_mask = "/data/integracar/amostras_car_mask/"
train_txt = "data-segments/train_sample.txt"
val_txt = "data-segments/validation_sample.txt"



from src.unet_run import UnetImporter

teste = UnetImporter(num_classes=1, loss_function="bce", optimizer="sgd")
model_name, 