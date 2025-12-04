from tqdm import tqdm
import glob
import rasterio
import numpy as np
import json
import os

output_tif = "/data/integracar/amostras_car_mask"

# Dicionário de cores (exemplo inicial, cores aproximadas)
legend_rgb = {
    (150, 150, 150): 0, #Afloramento Rochoso
    (251, 154, 153): 1, #Área Edificada
    (69, 175, 213): 2, #Brejo
    (150, 109, 207): 3,#Campo Rupestre/Altitude
    (128, 214, 16): 4, #Cultivo Agrícola - Abacaxi
    (247, 223, 8): 5, #Cultivo Agrícola - Banana
    (119, 9, 29): 6, #Cultivo Agrícola - Café
    (209, 163, 117): 7, #Cultivo Agrícola - Cana-De-Açúcar
    (231, 67, 97): 8, #Cultivo Agrícola - Coco-Da-Baía
    (245, 141, 23): 9, #Cultivo Agrícola - Mamão
    (55, 196, 201): 10, #Cultivo Agrícola - Outros Cultivos Permanentes
    (225, 175, 38): 11, #Cultivo Agrícola - Outros Cultivos Temporários
    (81, 77, 77): 12, #Extração Mineração
    (211, 127, 122): 13, #Macega
    (156, 68, 203): 14, #Mangue
    (133, 196, 221): 15, #Massa D'Água
    (13, 103, 19): 16, #Mata Nativa
    (51, 160, 44): 17, #Mata Nativa em Estágio Inicial de Regeneração
    (31, 205, 170): 18, #Outros
    (178, 214, 32): 19, #Pastagem
    (207, 103, 65): 20, #Reflorestamento - Eucalipto
    (243, 184, 129): 21, #Reflorestamento - Pinus
    (151, 132, 233): 22, #Reflorestamento - Seringueira
    (63, 231, 161): 23, #Restinga
    (245, 222, 193): 24 #Solo Exposto
}

# Dicionário reverso para salvar legenda legível
legend_names = {
    0: "Afloramento Rochoso",
    1: "Área Edificada",
    2: "Brejo",
    3: "Campo Rupestre/Altitude",
    4: "Cultivo Agrícola - Abacaxi",
    5: "Cultivo Agrícola - Banana",
    6: "Cultivo Agrícola - Café",
    7: "Cultivo Agrícola - Cana-De-Açúcar",
    8: "Cultivo Agrícola - Coco-Da-Baía",
    9: "Cultivo Agrícola - Mamão",
    10: "Cultivo Agrícola - Outros Cultivos Permanentes",
    11: "Cultivo Agrícola - Outros Cultivos Temporários",
    12: "Extração Mineração",
    13: "Macega",
    14: "Mangue",
    15: "Massa D'Água",
    16: "Mata Nativa",
    17: "Mata Nativa em Estágio Inicial de Regeneração",
    18: "Outros",
    19: "Pastagem",
    20: "Reflorestamento - Eucalipto",
    21: "Reflorestamento - Pinus",
    22: "Reflorestamento - Seringueira",
    23: "Restinga",
    24: "Solo Exposto"
}

mask_list = glob.glob('/data/integracar/amostras_car/*.tif')

for mask_path in tqdm(mask_list, desc="Processando imagens"):
    with rasterio.open(mask_path) as src:
        img = src.read()  # (C, H, W)
        profile = src.profile

    # Converter para (H, W, 3)
    img_rgb = np.transpose(img[:3], (1, 2, 0))

    # Criar raster vazio
    class_map = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

    # Converter cores → IDs
    for rgb, class_id in legend_rgb.items():
        mask = np.all(img_rgb == rgb, axis=-1)
        class_map[mask] = class_id

    # Atualizar perfil para 1 banda
    profile.update(dtype=rasterio.uint8, count=1)

    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    output_path = os.path.join(output_tif, f'{base_name}.tif')

    # Salvar raster classificado
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(class_map, 1)

    #print("✅ Raster classificado salvo em:", output_path)