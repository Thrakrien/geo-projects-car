import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, box
import processing
from qgis.core import (
QgsProject, QgsVectorLayer, QgsRasterLayer,
QgsVectorFileWriter, QgsWkbTypes,
QgsRectangle, QgsFeatureRequest
)

class GeospatialDataPreparation:
    """Classe para preparação de dados geoespaciais"""


    def __init__(self, project_dir):
        """
        Inicializa o preparador de dados

        Args:
            project_dir: Diretório raiz do projeto
        """
        self.project_dir = project_dir
        self.images_dir = os.path.join(project_dir, 'images')
        self.masks_dir = os.path.join(project_dir, 'masks')
        self.metadata_dir = os.path.join(project_dir, 'metadata')
        self.raw_dir = os.path.join(project_dir, 'raw')

        # Criar diretórios
        for directory in [self.images_dir, self.masks_dir,
            self.metadata_dir, self.raw_dir]:
            os.makedirs(directory, exist_ok=True)

    def load_wms_layer(self, url, layer_name, styles=''):
        """
        Carrega camada WMS
        Args:
            url: URL do servidor WMS
            layer_name: Nome da camada
            styles: Estilos (opcional)
        Returns:
            QgsRasterLayer
        """
        params = {
            'url': url,
            'layers': layer_name,
            'styles': styles,
            'format': 'image/png',
            'crs': 'EPSG:4326'
        }

        uri = f"url={params['url']}&amp;layers={params['layers']}&amp;" \
            f"styles={params['styles']}&amp;format={params['format']}&amp;" \
            f"crs={params['crs']}"

        wms_layer = QgsRasterLayer(uri, 'WMS Layer', 'wms')

        if wms_layer.isValid():
            QgsProject.instance().addMapLayer(wms_layer)
            print(f"Camada WMS '{layer_name}' carregada com sucesso!")

            return wms_layer

        else:
            print(f"Erro ao carregar camada WMS '{layer_name}'")
            return None
    
    def load_wfs_layer(self, url, typename):
        """
        Carrega camada WFS
        
        Args:
            url: URL do servidor WFS
            typename: Nome do tipo de feature
        Returns:
            QgsVectorLayer
        """
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typename': typename,
            'srsname': 'EPSG:4326'
            }

        import urllib.parse
        
        uri = url + '?' + urllib.parse.urlencode(params)
        wfs_layer = QgsVectorLayer(uri, "WFS Layer", "WFS")
        
        if wfs_layer.isValid():
            QgsProject.instance().addMapLayer(wfs_layer)
            print(f"Camada WFS '{typename}' carregada: "
            f"{wfs_layer.featureCount()} features")
            return wfs_layer
        else:
            print(f"Erro ao carregar camada WFS '{typename}'")
            return None

    def extract_layer_metadata(self, layer):
        """
        Extrai metadados de uma camada
        
        Args:
            layer: Camada QGIS (raster ou vetor)
        Returns:
            dict: Dicionário com metadados
        """
        metadata = {
            'name': layer.name(),
            'crs': layer.crs().authid(),
            'extent': {
                'xmin': layer.extent().xMinimum(),
                'ymin': layer.extent().yMinimum(),
                'xmax': layer.extent().xMaximum(),
                'ymax': layer.extent().yMaximum()
            }
        } 

        if layer.type() == 0: # Vetor
            metadata['type'] = 'vector'
            metadata['feature_count'] = layer.featureCount()
            metadata['geometry_type'] = layer.geometryType()
            metadata['fields'] = [field.name() for field in layer.fields()]

        elif layer.type() == 1: # Raster
            metadata['type'] = 'raster'
            metadata['width'] = layer.width()
            metadata['height'] = layer.height()
            metadata['band_count'] = layer.bandCount()
            metadata['pixel_size'] = {
            'x': layer.rasterUnitsPerPixelX(),
            'y': layer.rasterUnitsPerPixelY()
            }
        
        return metadata

    def export_vector_to_csv(self, vector_layer, output_path):
        """
        Exporta camada vetorial para CSV

        Args:
            vector_layer: Camada vetorial
            output_path: Caminho do arquivo CSV de saída
        """
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = 'CSV'
        save_options.fileEncoding = 'UTF-8'
        save_options.layerOptions = ['GEOMETRY=AS_WKT', 'SEPARATOR=COMMA']

        writer = QgsVectorFileWriter.create(
            fileName=output_path,
            fields=vector_layer.fields(),
            geometryType=QgsWkbTypes.NoGeometry,
            srs=vector_layer.crs(),
            transformContext=QgsProject.instance().transformContext(),
            options=save_options
        )

        if writer.hasError() == QgsVectorFileWriter.NoError:
            for feature in vector_layer.getFeatures():
                writer.addFeature(feature)
            del writer
            print(f"CSV exportado: {output_path}")
        else:
            print(f"Erro ao exportar CSV: {writer.errorMessage()}")

    def create_segmentation_mask_pyqgis(self, vector_layer, reference_raster, output_path, class_field='classe'):
        """
        Cria máscara de segmentação usando PyQGIS

        Args:
            vector_layer: Camada vetorial com polígonos anotados
            reference_raster: Raster de referência para dimensões
            output_path: Caminho de saída da máscara
            class_field: Nome do campo com valores de classe
        """
        params = {
            'INPUT': vector_layer,
            'FIELD': class_field,
            'UNITS': 1,
            'WIDTH': reference_raster.width(),
            'HEIGHT': reference_raster.height(),
            'EXTENT': reference_raster.extent(),
            'NODATA': 0,
            'DATA_TYPE': 0, # Byte
            'INIT': 0,
            'OUTPUT': output_path
        }

        try:
            result = processing.run("gdal:rasterize", params)
            print(f"Máscara criada: {output_path}")
            return result['OUTPUT']
        except Exception as e:
            print(f"Erro ao criar máscara: {e}")
            return None

    def create_segmentation_mask_rasterio(self, shapefile_path, reference_tif_path, output_path, class_column='classe'):
        """
        Cria máscara de segmentação usando Rasterio

        Args:
            shapefile_path: Caminho do shapefile com polígonos
            reference_tif_path: Caminho do TIF de referência
            output_path: Caminho de saída da máscara
            class_column: Nome da coluna com valores de classe
        """
        
        # Carregar referência
        with rasterio.open(reference_tif_path) as src:
            meta = src.meta.copy()
            transform = src.transform
            shape = (src.height, src.width)
            crs = src.crs

        # Carregar shapefile
        gdf = gpd.read_file(shapefile_path)

        # Reprojetar se necessário
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        
        # Criar shapes com valores de classe
        shapes = [(mapping(geom), value) for geom, value in zip(gdf.geometry, gdf[class_column])]
        
        # Rasterizar
        mask = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype='uint8'
            )

        # Atualizar metadados
        meta.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
            })
        
        # Salvar
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask, 1)
        print(f"Máscara criada (rasterio): {output_path}")

    def download_tiles_from_csv(self, csv_path, wms_layer, output_dir):
        """
        Baixa tiles baseado em CSV com coordenadas

        Args:
            csv_path: Caminho do CSV com coordenadas
            wms_layer: Camada WMS/raster para recortar
            output_dir: Diretório de saída
        """
        
        # Carregar CSV
        df = pd.read_csv(csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, row in df.iterrows():
            tile_id = row['id']
            extent = QgsRectangle(
                row['xmin'], row['ymin'],
                row['xmax'], row['ymax']
                )
        
        output_path = os.path.join(output_dir, f'tile_{tile_id}.tif')
        
        params = {
            'INPUT': wms_layer,
            'PROJWIN': f"{extent.xMinimum()},{extent.xMaximum()},"
            f"{extent.yMinimum()},{extent.yMaximum()}",
            'OUTPUT': output_path
            }
        
        try:
            processing.run("gdal:cliprasterbyextent", params)
            print(f"Tile {tile_id} baixado")
        except Exception as e:
            print(f"Erro no tile {tile_id}: {e}")

    def clip_raster_by_features(self, raster_layer, vector_layer, output_dir, id_field='id'):
        """
        Recorta raster baseado em features vetoriais

        Args:
            raster_layer: Camada raster
            vector_layer: Camada vetorial com áreas de interesse
            output_dir: Diretório de saída
            id_field: Campo com ID único
        """

        os.makedirs(output_dir, exist_ok=True)
        for feature in vector_layer.getFeatures():
            fid = feature[id_field]
            extent = feature.geometry().boundingBox()

            output_path = os.path.join(output_dir, f'clip_{fid}.tif')

            params = {
                'INPUT': raster_layer,
                'PROJWIN': f"{extent.xMinimum()},{extent.xMaximum()},"
                f"{extent.yMinimum()},{extent.yMaximum()}",
                'OUTPUT': output_path
            }

            try:
                processing.run("gdal:cliprasterbyextent", params)
                print(f"Recorte criado: clip_{fid}.tif")
            except Exception as e:
                print(f"Erro ao recortar feature {fid}: {e}")

    def batch_create_masks(self, annotations_layer, reference_raster, output_dir, image_ids):
        """
        Cria máscaras em lote para múltiplas imagens
        
        Args:
            annotations_layer: Camada com todas as anotações
            reference_raster: Raster de referência
            output_dir: Diretório de saída das máscaras
            image_ids: Lista de IDs de imagens
        """
        
        os.makedirs(output_dir, exist_ok=True)

        for img_id in image_ids:
            # Filtrar anotações para esta imagem
            expression = f'"image_id" = {img_id}'
            annotations_layer.setSubsetString(expression)

            if annotations_layer.featureCount() > 0:
                mask_path = os.path.join(
                    output_dir,
                    f'image_{img_id:03d}_mask.tif')

                self.create_segmentation_mask_pyqgis(
                    annotations_layer,
                    reference_raster,
                    mask_path,
                    class_field='classe'
                    )
            else:
                print(f"Sem anotações para imagem {img_id}")
            
        # Limpar filtro
        annotations_layer.setSubsetString('')
        print("Criação de máscaras em lote concluída!")
        
    def create_class_mapping_file(self, class_dict, output_path):
        """
        Cria arquivo JSON com mapeamento de classes

        Args:
            class_dict: Dicionário {valor: nome_classe}
            output_path: Caminho do arquivo JSON
        """
        mapping = {
            'class_mapping': class_dict,
            'num_classes': len(class_dict)
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"Arquivo de mapeamento criado: {output_path}")

     