# Geo Projects CAR

Pipeline de segmentacao semantica de imagens para o projeto de mestrado. O fluxo atual usa PyTorch, `segmentation-models-pytorch`, dataset customizado, patchify/unpatchify, metricas de IoU e visualizacao de predicoes.

## Estrutura Principal

- `train_code.py`: script principal de treino, validacao, inferencia em imagens de exemplo e logging.
- `src/dataset.py`: dataset customizado e criacao dos dataloaders com patchify.
- `src/inference.py`: inferencia por sliding window e agregacao espacial dos logits.
- `src/models.py`: factory/registry para trocar arquiteturas sem alterar o pipeline.
- `src/training_config.py`: config padrao, leitura de YAML/JSON e validacoes de patch grid e splits.
- `configs/experiments/`: configs de experimentos.
- `experiments/`: saida dos treinamentos, checkpoints, metricas e figuras.

## Instalar Dependencias

Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instale as dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Se usar CUDA, confirme que a versao do PyTorch instalada e compativel com a sua GPU. Em alguns ambientes, pode ser necessario instalar o PyTorch seguindo o comando recomendado no site oficial do PyTorch antes de instalar o restante das dependencias.

## Configurar Experimentos

As configs principais ficam em:

```text
configs/experiments/unet_baseline.yaml
configs/experiments/deeplabv3plus_baseline.yaml
```

Para trocar o modelo, altere apenas os campos de modelo na config:

```yaml
architecture: unet
encoder_name: efficientnet-b5
encoder_weights: imagenet
in_channels: 3
num_classes: 5
```

Exemplo para DeepLabV3+:

```yaml
architecture: deeplabv3plus
encoder_name: efficientnet-b5
encoder_weights: imagenet
in_channels: 3
num_classes: 5
```

O pipeline espera que o modelo retorne logits crus no formato `[B, C, H, W]`. Por isso, a factory cria os modelos com `activation=None`, mantendo compatibilidade com `CrossEntropyLoss`.

## Patchify e Sobreposicao

A consistencia espacial da reconstrucao depende de `image_size`, `patch_size`, `patch_step` e `inference_stride`.

Treino sem sobreposicao:

```yaml
image_size: 2048
patch_size: 512
patch_step: 512
```

Inferencia com 50% de sobreposicao:

```yaml
patch_size: 512
inference_overlap: 256
```

`inference_overlap: 256` gera automaticamente `inference_stride: 256`. Se preferir controlar diretamente o stride, use:

```yaml
inference_stride: 256
```

Para treino com patches sobrepostos, tambem e possivel usar:

```yaml
patch_overlap: 128
```

Isso gera `patch_step = patch_size - patch_overlap`. A config valida se a grade cobre a imagem de forma consistente, evitando gaps na reconstrucao.

## Rodar Treinamento

Ative o ambiente:

```bash
source venv/bin/activate
```

Rodar com a config padrao em Python:

```bash
PYTHONPATH=. python train_code.py
```

Rodar UNet com YAML:

```bash
PYTHONPATH=. python train_code.py --config configs/experiments/unet_baseline.yaml
```

Rodar DeepLabV3+ com YAML:

```bash
PYTHONPATH=. python train_code.py --config configs/experiments/deeplabv3plus_baseline.yaml
```

As saidas sao salvas em `experiments/<nome_do_experimento>_<timestamp>/`, incluindo:

- `config.json`
- `metrics.csv`
- `best_model.pth`
- `loss_weights.npy`, quando `use_class_weights: true`
- figuras de resumo e overlays
- `summary.json`

## Rodar em Segundo Plano com nohup

Crie uma pasta para logs, se ainda nao existir:

```bash
mkdir -p logs
```

Rodar UNet em segundo plano:

```bash
nohup bash -lc 'source venv/bin/activate && PYTHONPATH=. python train_code.py --config configs/experiments/unet_baseline.yaml' > logs/unet_baseline.log 2>&1 &
```

Rodar DeepLabV3+ em segundo plano:

```bash
nohup bash -lc 'source venv/bin/activate && PYTHONPATH=. python train_code.py --config configs/experiments/deeplabv3plus_baseline.yaml' > logs/deeplabv3plus_baseline.log 2>&1 &
```

Verificar o log durante a execucao:

```bash
tail -f logs/unet_baseline.log
```

Listar processos relacionados ao treino:

```bash
ps aux | grep train_code.py
```

Encerrar um processo pelo PID:

```bash
kill <PID>
```

## Validacao Rapida

Rodar testes de config espacial e factory de modelos:

```bash
PYTHONPATH=. venv/bin/pytest tests/test_training_config.py tests/test_models.py
```

Checar sintaxe dos arquivos principais:

```bash
PYTHONPATH=. venv/bin/python -m py_compile train_code.py src/models.py src/training_config.py
```

## Cuidados para Comparar Modelos

Para uma comparacao justa entre arquiteturas:

- mantenha os mesmos arquivos `train_txt` e `val_txt`;
- mantenha os mesmos `image_size`, `patch_size`, `patch_step` e `inference_stride`;
- use as mesmas transforms e normalizacao;
- use a mesma loss, `ignore_index` e pesos de classe;
- use o mesmo encoder e `encoder_weights` quando quiser comparar apenas a arquitetura;
- registre diferencas de batch size, GPU e tempo de treino;
- evite data leakage: a config verifica nomes repetidos entre treino e validacao, mas ainda vale conferir se patches de uma mesma cena geografica nao aparecem em splits diferentes.
