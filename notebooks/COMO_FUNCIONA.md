# Documentação dos Notebooks — Projeto CAR (Segmentação de Uso do Solo)

Este documento explica o funcionamento de cada notebook do projeto, o que ele faz, e como usar passo a passo.

Todos os notebooks trabalham com imagens georeferenciadas `.tif` de uso do solo, usando **25 classes** de cobertura terrestre com cores RGB específicas.

---

## Pré-requisitos comuns

### Pastas necessárias

Os notebooks que fazem análise visual precisam de **duas pastas** com as mesmas imagens (mesmo nome de arquivo):

- **Pasta de máscaras coloridas**: cada pixel tem a cor RGB da classe de uso do solo
- **Pasta de imagens de satélite**: imagens puras sem classificação

Exemplo:
```
D:\Imagens CAR - Testes 11-02 COM COR\amostra_13.tif   ← máscara colorida
D:\Imagens CAR - Testes 11-02 SEM COR\amostra_13.tif   ← satélite puro
```

### Dependências

```
pip install numpy rasterio matplotlib scipy tqdm
```

---

## Índice

1. [qualitative_analysis.ipynb](#1-qualitative_analysisipynb) — Análise qualitativa por classe (grids visuais)
2. [class_analysis.ipynb](#2-class_analysisipynb) — Análise detalhada de uma classe específica
3. [mask_converter.ipynb](#3-mask_converteripynb) — Conversão de máscaras RGB para IDs numéricos
4. [count_classes.ipynb](#4-count_classesipynb) — Contagem de pixels por classe
5. [Tabela de classes](#5-tabela-de-classes)

---

## 1. qualitative_analysis.ipynb

### O que faz

Gera uma **análise qualitativa visual** de todas as 25 classes. Cada célula do notebook é responsável por **uma classe**, exibindo um grid com **todas** as imagens onde aquela classe aparece, organizadas em fileiras de 5 colunas.

As imagens são ordenadas por ocupação (maior primeiro), mostrando a classe destacada sobre a imagem de satélite com borda vermelha e overlay branco semi-transparente.

**Não salva nada em disco** — tudo é exibido apenas no notebook.

### Estrutura do notebook

| Célula | Tipo | Conteúdo |
|--------|------|----------|
| 0 | Markdown | Título e explicação |
| 1 | Código | Imports, configuração de caminhos e dicionário das 25 classes |
| 2 | Código | Funções auxiliares (find_class_in_mask, load_raw_image, highlight_on_raw, calc_occupation) |
| 3 | Código | **Indexação** — escaneia todas as imagens uma vez e mapeia quais classes aparecem em cada uma |
| 4 | Código | Função `show_class_grid()` — gera o grid de uma classe |
| 5 | Markdown | Separador |
| 6–30 | Código | **Uma célula por classe** (25 classes), cada uma chama `show_class_grid("Nome da Classe")` |

### Passo a passo de como usar

1. **Abra o notebook** no Cursor/VS Code/Jupyter
2. **Ajuste os caminhos na Célula 1**:
   ```python
   MASKS_DIR  = r"D:\Imagens CAR - Testes 11-02 COM COR"   # máscaras coloridas
   RAW_DIR    = r"D:\Imagens CAR - Testes 11-02 SEM COR"    # satélite puro
   N_COLS     = 5      # colunas por fileira
   MAX_IMAGES = None   # None = todas; ou um número pra limitar (ex: 30)
   ```
3. **Execute as Células 1 a 4** (configuração, funções, indexação e definição do grid)
   - A indexação (Célula 3) leva ~25 segundos para 281 imagens. Precisa rodar só uma vez.
4. **Execute qualquer célula de classe** (6 a 30) para ver o grid daquela classe
   - Pode rodar todas de uma vez com "Run All" ou uma por uma
5. **Se ficar pesado**: defina `MAX_IMAGES = 30` (ou outro número) na Célula 1 e re-execute

### Funções auxiliares

**`find_class_in_mask(tif_path, target_rgb)`**
- Abre a máscara TIF, compara cada pixel com a cor RGB da classe
- Retorna a máscara booleana (True onde a classe aparece) e a imagem RGB

**`load_raw_image(raw_path)`**
- Abre a imagem de satélite pura
- Retorna como array RGB (altura, largura, 3)

**`highlight_on_raw(raw_rgb, mascara)`**
- Aplica overlay branco semi-transparente (alpha=0.15) sobre a área da classe
- Desenha borda vermelha ao redor usando dilatação morfológica
- Retorna a imagem final destacada

**`calc_occupation(mascara)`**
- Calcula a porcentagem de pixels da classe em relação ao total

**`show_class_grid(class_name, n_cols, max_images)`**
- Cria um grid de `n_cols` colunas × N fileiras para exibir todas as imagens onde a classe aparece
- Imagens ordenadas por ocupação (maior primeiro)
- Título mostra: nome da classe, RGB, quantidade e % de imagens onde aparece
- `plt.close(fig)` libera memória após renderizar

### Como exportar para PDF

1. Execute todas as células do notebook
2. `Ctrl + Shift + P` → "Jupyter: Export to HTML"
3. Escolha onde salvar o `.html`
4. Abra o HTML no navegador → `Ctrl + P` → Salvar como PDF

---

## 2. class_analysis.ipynb

### O que faz

Faz uma **análise detalhada de uma única classe** por vez. Percorre todas as imagens buscando a classe escolhida, gera visualizações, calcula estatísticas (aparição, ocupação, coordenadas geográficas) e **salva tudo em disco** (imagens destacadas em `.png` + estatísticas e coordenadas em `.txt`).

### Estrutura do notebook

| Célula | Tipo | Conteúdo |
|--------|------|----------|
| 0 | Markdown | Título e instruções |
| 1 | Código | Configuração: classe alvo, caminhos, listagem de arquivos |
| 2 | Código | Funções auxiliares (5 funções) |
| 3 | Markdown | Separador "Processamento" |
| 4 | Código | Loop principal: processa cada imagem, destaca, salva |
| 5 | Markdown | Separador "Estatísticas" |
| 6 | Código | Calcula e exibe estatísticas + gráfico de barras |
| 7 | Código | Salva arquivos `.txt` com estatísticas e coordenadas |

### Passo a passo de como usar

1. **Ajuste a Célula 1**:
   ```python
   TARGET_CLASS_NAME = "Mata Nativa"   # troque para a classe desejada
   MASKS_DIR  = r"D:\Imagens CAR - Testes 11-02 COM COR"
   RAW_DIR    = r"D:\Imagens CAR - Testes 11-02 SEM COR"
   OUTPUT_DIR = r"D:\Imagens CAR - Testes Notebook 11-02"
   ```
2. **Execute todas as células** (Run All)
3. **Aguarde o processamento** (Célula 4): percorre todas as imagens, exibe em grupos de 3 e salva cada uma
4. **Veja as estatísticas** (Célula 6): aparição, ocupação média/max/min, gráfico de barras, tabela com coordenadas
5. **Resultados salvos em disco**:
   - `OUTPUT_DIR/highlighted/highlighted_{nome}.png` — imagens com a classe destacada
   - `OUTPUT_DIR/estatisticas_{classe}.txt` — dados gerais
   - `OUTPUT_DIR/coordenadas_{classe}.txt` — centróides e bounding boxes geográficos

### Diferenças para o qualitative_analysis.ipynb

| Aspecto | class_analysis | qualitative_analysis |
|---------|---------------|---------------------|
| Classes por execução | **1** (muda `TARGET_CLASS_NAME`) | **Todas 25** de uma vez |
| Salva em disco | **Sim** (imagens .png + .txt) | **Não** (só exibe) |
| Coordenadas geográficas | **Sim** (centróide + bbox) | **Não** |
| Estatísticas detalhadas | **Sim** (gráfico + tabela) | Resumo no título do grid |
| Objetivo | Análise profunda de 1 classe | Visão geral qualitativa de todas |

### Funções auxiliares

As mesmas do `qualitative_analysis.ipynb`, com adição de:

**`get_coordinates(meta, mascara)`**
- Converte posição dos pixels da classe para coordenadas geográficas reais usando o `transform` do rasterio
- Calcula o centróide (ponto médio) e o bounding box (retângulo que engloba a área)
- Retorna dicionário com centroid_x, centroid_y, bbox e CRS

### Para analisar outra classe

1. Mude `TARGET_CLASS_NAME` na Célula 1 (ex: `"Pastagem"`, `"Macega"`, `"Solo Exposto"`)
2. Rode todas as células de novo
3. Os arquivos de saída têm o nome da classe, então **não sobrescrevem** os anteriores

---

## 3. mask_converter.ipynb

### O que faz

Converte as **máscaras coloridas RGB** (3 bandas, cada cor = uma classe) para **máscaras de canal único** (1 banda, cada pixel = ID numérico da classe). Isso é necessário para treinar modelos de segmentação, que esperam labels numéricos.

O notebook oferece **3 tipos de conversão** diferentes.

### Estrutura do notebook

| Célula | Conteúdo |
|--------|----------|
| 0 | Instalação do rasterio |
| 1 | **Conversão agrupada (14 classes)**: agrupa as 25 classes em 14 categorias mais amplas (ex: todos cultivos viram "Áreas de Cultivo", todos reflorestamentos viram "Reflorestamento") |
| 2 | Verificação do resultado |
| 3 | **Função `convert_pixels()`**: suporta conversão "Full" (25 classes, IDs 0-24) e "binary_forest" (3 classes: sem_vegetação, com_vegetação, nulos) |

### Tipos de conversão

#### Agrupada (Célula 1) — 14 classes

Agrupa classes similares para reduzir complexidade:

| ID | Classe agrupada | Classes originais incluídas |
|----|----------------|----------------------------|
| 0 | Afloramento/Edificada/Brejo | Afloramento Rochoso, Área Edificada, Brejo |
| 3 | Campo Rupestre | Campo Rupestre/Altitude |
| 4 | Áreas de Cultivo | Todos os 8 cultivos agrícolas |
| 5 | Extração Mineração | Extração Mineração |
| 6 | Macega | Macega |
| 7 | Mangue | Mangue |
| 8 | Massa D'Água | Massa D'Água |
| 9 | Mata Nativa | Mata Nativa + Estágio Inicial de Regeneração |
| 10 | Outros | Outros |
| 11 | Pastagem/Solo Exposto | Pastagem + Solo Exposto |
| 12 | Reflorestamento | Eucalipto + Pinus + Seringueira |
| 13 | Restinga | Restinga |

#### Full (Célula 3, tipo "Full") — 25 classes

Cada classe original recebe um ID único de 0 a 24. Sem agrupamento.

#### Binária (Célula 3, tipo "binary_forest") — 3 classes

| ID | Classe |
|----|--------|
| 0 | Sem vegetação (rochoso, edificada, mineração, água, outros, solo exposto) |
| 1 | Com vegetação (cultivos, macega, mangue, mata, pastagem, reflorestamento, restinga) |
| 2 | Nulos (linhas brancas na máscara) |

### Passo a passo de como usar

1. **Ajuste o caminho de entrada** (onde estão as máscaras RGB):
   ```python
   mask_list = glob.glob('/data/integracar/amostras_car/*.tif')
   ```
2. **Ajuste o caminho de saída**:
   ```python
   output_tif = "/data/integracar/amostras_car_mask/"
   ```
3. **Escolha o tipo de conversão**:
   - Para 14 classes agrupadas: execute a **Célula 1**
   - Para 25 classes ou binária: execute a **Célula 3** alterando o parâmetro:
     ```python
     legend_names, legend_rgb = convert_pixels(convert_type="Full")        # 25 classes
     legend_names, legend_rgb = convert_pixels(convert_type="binary_forest")  # 3 classes
     ```
4. **Execute** — processa todas as imagens e salva os TIFs convertidos na pasta de saída

### O que salva

- Um arquivo `.tif` por máscara na pasta de saída, com 1 banda e valores uint8 (IDs das classes)

### ⚠️ Observação importante

As 3 formas de conversão usam mapeamentos de IDs **diferentes**. É essencial usar o mesmo mapeamento no treinamento do modelo e na avaliação. Os IDs usados aqui devem bater com o `num_classes` do modelo.

---

## 4. count_classes.ipynb

### O que faz

Conta quantos **pixels** de cada classe existem no dataset inteiro. Percorre todas as máscaras coloridas e compara cada pixel com as 25 cores RGB, gerando uma tabela de distribuição.

### Passo a passo de como usar

1. **Ajuste o caminho das máscaras**:
   ```python
   mask_list = glob.glob('/data/integracar/amostras_car/*.tif')
   ```
2. **Execute todas as células**
3. **Resultado**: tabela com nome da classe, quantidade de pixels e porcentagem

### Resultado típico

As classes mais dominantes são:
- **Pastagem** (~33%)
- **Mata Nativa** (~16%)
- **Cultivo Agrícola - Café** (~15%)

Classes raras como Mangue, Abacaxi e Campo Rupestre representam menos de 1% cada.

### ⚠️ Observação

Este notebook usa IDs de **1 a 25** (diferente do `mask_converter.ipynb` que usa 0-24). Isso não afeta a contagem pois ela é feita por cor RGB, mas é uma inconsistência a ser observada.

---

## 5. Tabela de classes

As 25 classes de uso do solo com suas cores RGB:

| # | Classe | Cor RGB |
|---|--------|---------|
| 0 | Afloramento Rochoso | (150, 150, 150) |
| 1 | Área Edificada | (251, 154, 153) |
| 2 | Brejo | (69, 175, 213) |
| 3 | Campo Rupestre/Altitude | (150, 109, 207) |
| 4 | Cultivo Agrícola - Abacaxi | (128, 214, 16) |
| 5 | Cultivo Agrícola - Banana | (247, 223, 8) |
| 6 | Cultivo Agrícola - Café | (119, 9, 29) |
| 7 | Cultivo Agrícola - Cana-De-Açúcar | (209, 163, 117) |
| 8 | Cultivo Agrícola - Coco-Da-Baía | (231, 67, 97) |
| 9 | Cultivo Agrícola - Mamão | (245, 141, 23) |
| 10 | Cultivo Agrícola - Outros Cultivos Permanentes | (55, 196, 201) |
| 11 | Cultivo Agrícola - Outros Cultivos Temporários | (225, 175, 38) |
| 12 | Extração Mineração | (81, 77, 77) |
| 13 | Macega | (211, 127, 122) |
| 14 | Mangue | (156, 68, 203) |
| 15 | Massa D'Água | (133, 196, 221) |
| 16 | Mata Nativa | (13, 103, 19) |
| 17 | Mata Nativa em Estágio Inicial de Regeneração | (51, 160, 44) |
| 18 | Outros | (31, 205, 170) |
| 19 | Pastagem | (178, 214, 32) |
| 20 | Reflorestamento - Eucalipto | (207, 103, 65) |
| 21 | Reflorestamento - Pinus | (243, 184, 129) |
| 22 | Reflorestamento - Seringueira | (151, 132, 233) |
| 23 | Restinga | (63, 231, 161) |
| 24 | Solo Exposto | (245, 222, 193) |

---

## Resumo rápido

| Notebook | Objetivo | Salva em disco? | Classes |
|----------|----------|-----------------|---------|
| `qualitative_analysis.ipynb` | Grid visual de todas as classes | Não | Todas 25 |
| `class_analysis.ipynb` | Análise profunda de 1 classe | Sim (imagens + txt) | 1 por vez |
| `mask_converter.ipynb` | Converter RGB → ID numérico | Sim (TIFs convertidos) | 14, 25 ou 3 |
| `count_classes.ipynb` | Contar pixels por classe | Sim (txt) | Todas 25 |
