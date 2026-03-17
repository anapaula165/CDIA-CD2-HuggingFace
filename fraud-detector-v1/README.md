---
pipeline_tag: tabular-classification
tags:
  - fraud-detection
  - scikit-learn
  - random-forest
  - tabular-data
language:
  - pt
---

# fraud-detector-v1

Modelo de classificacao binaria para deteccao de fraude em transacoes financeiras. O projeto usa um `RandomForestClassifier` treinado sobre um dataset tabular sintetico com sinais simples de risco.

## Objetivo

Prever se uma transacao deve ser classificada como `legitimo` ou `fraude`.

## Features do dominio

As features usadas pelo modelo representam sinais basicos de fraude no dominio financeiro:

- `valor_transacao`: valor monetario da operacao
- `hora_transacao`: hora em que a transacao ocorreu
- `distancia_ultima_compra`: distancia estimada em relacao a ultima compra
- `tentativas_senha`: numero de tentativas de senha antes da confirmacao
- `pais_diferente`: indica se a transacao ocorreu em pais diferente do padrao esperado

## Treinamento

- Dataset: 2000 amostras sinteticas
- Split: 80% treino / 20% teste com `stratify=y`
- Modelo: `RandomForestClassifier(n_estimators=100, random_state=42)`

## Metricas reais do Bloco 2

Metricas obtidas pela execucao real do arquivo [`main.py`](/c:/Users/berna/Desktop/jeff/Bedimand/fraud-detector-v1/main.py):

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| legitimo | 1.00 | 1.00 | 1.00 | 276 |
| fraude | 1.00 | 1.00 | 1.00 | 124 |
| accuracy | - | - | 1.00 | 400 |
| macro avg | 1.00 | 1.00 | 1.00 | 400 |
| weighted avg | 1.00 | 1.00 | 1.00 | 400 |

Artefato gerado:

- `model.pkl`: 63.7 KB

## Limitacoes do modelo

- O dataset e sintetico, entao as metricas nao representam desempenho real em producao.
- As classes foram geradas com padroes muito separados, o que facilita demais a tarefa e explica o resultado perfeito.
- O modelo usa poucas variaveis e nao considera contexto historico mais rico, como dispositivo, estabelecimento, frequencia de compra ou comportamento do usuario ao longo do tempo.
- Nao ha calibracao de probabilidade, ajuste de limiar de decisao ou analise de custo de falso positivo versus falso negativo.
- O modelo nao foi validado contra drift de dados, ataques adversariais ou dados reais desbalanceados.

## Quando usar

Este modelo serve bem como demonstracao educacional de pipeline de treinamento, serializacao e publicacao de modelo tabular. Para uso real, seria necessario treinar e validar com dados reais, revisar features e implementar monitoramento.
