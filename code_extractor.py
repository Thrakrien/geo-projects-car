from qgis.core import (QgsProject, QgsRectangle, QgsMapSettings, QgsVectorLayer,
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsGeometry)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface
import os

# 1. Nomes das camadas EXATAMENTE como aparecem no Painel de Camadas do QGIS
NOME_CAMADA_PONTOS = 'amostras_car'
NOME_CAMADA_WMS = 'uso_do_solo'

# 2. Pastas de destino para salvar as imagens
PASTA_IMAGENS = '/mnt/amostras_car'

# 3. Definição dos Sistemas de Coordenadas (SRC/CRS)
#    Verifique o SRC da sua camada de pontos (clique direito > Propriedades > Fonte)
SRC_PONTOS = 'EPSG:31984'        # Provavelmente SIRGAS 2000 em graus (ou EPSG:4326 para WGS84)
SRC_DESTINO = 'EPSG:31984'      # SIRGAS 2000 / UTM Zone 24S (para sua região em metros)

# 4. Tamanho do recorte em metros
LARGURA_RECORTE_METROS = 1000
ALTURA_RECORTE_METROS = 1000

# 5. Resolução da imagem de saída em pixels
LARGURA_IMAGEM_PIXELS = 1024
ALTURA_IMAGEM_PIXELS = 1024

# 6. Nome do campo na camada de pontos para usar no nome do arquivo
CAMPO_NOME_ARQUIVO = 'cod_imovel'

### ====================================================================== ###
### FIM DOS PARÂMETROS ###
### ====================================================================== ###


# --- FUNÇÃO AUXILIAR PARA CRIAR WORLD FILE MANUALMENTE ---
def create_world_file(map_settings, image_path):
    ext = map_settings.extent()
    size = map_settings.outputSize()
    pixel_width = ext.width() / size.width()
    line_2, line_3 = 0.0, 0.0
    pixel_height = -ext.height() / size.height()
    x_coord = ext.xMinimum() + (pixel_width / 2.0)
    y_coord = ext.yMaximum() + (pixel_height / 2.0)
    path_sem_ext, _ = os.path.splitext(image_path)
    tfw_path = path_sem_ext + '.tfw'
    with open(tfw_path, 'w') as f:
        f.write(f"{pixel_width}\n{line_2}\n{line_3}\n{pixel_height}\n{x_coord}\n{y_coord}\n")

# --- SCRIPT PRINCIPAL ---
def run_dataset_creation():
    project = QgsProject.instance()

    try:
        camada_pontos = project.mapLayersByName(NOME_CAMADA_PONTOS)[0]
        camada_wms = project.mapLayersByName(NOME_CAMADA_WMS)[0]
    except IndexError:
        iface.messageBar().pushMessage("Erro Crítico", "Camada de pontos ou WMS não encontrada. Verifique os NOMES!", level=3, duration=7)
        return
        
    os.makedirs(PASTA_IMAGENS, exist_ok=True)
    
    # Prepara a transformação de coordenadas
    src_crs = QgsCoordinateReferenceSystem(SRC_PONTOS)
    dest_crs = QgsCoordinateReferenceSystem(SRC_DESTINO)
    transformacao = QgsCoordinateTransform(src_crs, dest_crs, project.transformContext())

    print(">>> Iniciando a geração de ortofotos...")
    total_features = camada_pontos.featureCount()
    
    for i, feature in enumerate(camada_pontos.getFeatures()):
        print(f"--- Processando ponto {i + 1} de {total_features} ---")
        
        # --- LÓGICA DE TRANSFORMAÇÃO DE COORDENADAS ---
        geom_original = feature.geometry()
        geom_transformada = QgsGeometry(geom_original) # Cria uma cópia para transformar
        geom_transformada.transform(transformacao)
        ponto_central_em_metros = geom_transformada.asPoint()
        # ------------------------------------------------
        
        # Usa o ponto transformado (em metros) para criar a extensão
        extensao = QgsRectangle.fromCenterAndSize(ponto_central_em_metros, LARGURA_RECORTE_METROS, ALTURA_RECORTE_METROS)
        
        config_mapa = QgsMapSettings()
        config_mapa.setDestinationCrs(dest_crs) 
        config_mapa.setLayers([camada_wms])
        config_mapa.setExtent(extensao)
        config_mapa.setOutputSize(QSize(LARGURA_IMAGEM_PIXELS, ALTURA_IMAGEM_PIXELS))
        config_mapa.setBackgroundColor(QColor(255, 255, 255, 0))
        
        if CAMPO_NOME_ARQUIVO and CAMPO_NOME_ARQUIVO in feature.fields().names():
            nome_base = feature[CAMPO_NOME_ARQUIVO]
        else:
            nome_base = f'amostra_{i + 1}'
            
        caminho_imagem = os.path.join(PASTA_IMAGENS, f'{nome_base}.tif')
        
        renderizador = QgsMapRendererParallelJob(config_mapa)
        renderizador.start()
        renderizador.waitForFinished()
        img = renderizador.renderedImage()
        img.save(caminho_imagem, "tif")
        
        create_world_file(config_mapa, caminho_imagem)
        
        print(f"SALVO: Imagem '{nome_base}.tif'")

    iface.messageBar().pushMessage("Sucesso!", f"Geração de imagens concluída com {total_features} amostras.", level=1, duration=7)
    print(">>> Processo concluído!")

# Executa a função principal
run_dataset_creation()