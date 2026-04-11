# AGENTS.md

## Objetivo do projeto
Projeto de mestrado focado em segmentação semântica de imagens.

## Prioridades
1. Correção espacial da segmentação
2. Reprodutibilidade
3. Modularidade
4. Legibilidade
5. Mudanças mínimas por etapa

## Convenções
- usar type hints
- seguir PEP 8
- docstrings no estilo Google ou NumPy
- evitar funções com múltiplas responsabilidades
- evitar hardcode de caminhos
- centralizar hiperparâmetros em config
- separar treino, inferência, visualização e utilitários

## Regras para o agente
- não alterar múltiplos módulos de uma vez sem necessidade;
- sempre propor plano antes de refatorações grandes;
- preservar comportamento atual;
- explicar toda mudança relevante;
- sugerir testes antes de mudanças arriscadas;
- sinalizar qualquer risco de data leakage;
- verificar coerência entre patchify, transforms e unpatchify.

## Validação
- rodar testes unitários quando existirem
- validar shapes de entrada e saída
- validar reconstrução espacial
- validar métricas básicas após mudanças