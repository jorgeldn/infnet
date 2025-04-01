<img src="docs\images\infnet-logo.png" width="200">


# Projeto da Disciplina de Engenharia de Machine Learning

Link do projeto:
[https://github.com/jorgeldn/infnet](https://github.com/jorgeldn/infnet)

## Overview

Desenvolver um preditor de arremessos usando duas abordagens (regressão e classificação) para prever se o "Black Mamba" (apelido de Kobe) acertou ou errou a cesta.
Na pasta `data/01_raw` estão disposiveis os arquivos: **dataset_kobe_dev.parquet** e **dataset_kobe_prod.parquet** alvos de estudo deste projeto.

## Dependências

Para executar o projeto, foi criado um ambiente virtual exclusivo a partir do comando:

```
python -m venv venv-pos-ia
```

Foi gerado um arquivo `requirements.in` com as dependências do projeto.
O arquivo foi compilado com o comando `pip-compile requirements.in` e gerou um arquivo `requirements.txt`.

Para instalação, executar:

```
pip install -r requirements.txt
```

## Questão 1: Estrutura e solução do projeto?

Este projeto foi desenvolvido utilizando o framework Kedro versão `kedro 0.19.11`.

<img src="docs\images\folders-structure.png" width="200">

## Questão 2: Diagrama de etapas do projeto

<img src="docs\images\diagram.png">

## Questão 3: Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines?

### 1. **Rastreamento de Experimentos**
   - **MLflow**: Permite registrar e acompanhar diferentes experimentos de modelagem, armazenando métricas, hiperparâmetros e artefatos de cada execução. Isso facilita a comparação entre abordagens de regressão e classificação.
   - **PyCaret**: Automatiza a experimentação com diferentes modelos, armazenando métricas e permitindo rápida comparação entre técnicas.

### 2. **Funções de Treinamento**
   - **Scikit-Learn**: Fornece bibliotecas para treinamento, validação e ajuste de modelos preditivos, incluindo algoritmos de regressão e classificação.
   - **PyCaret**: Facilita a seleção automática dos melhores modelos, aplicando técnicas como AutoML e tunagem de hiperparâmetros sem necessidade de configuração manual.

### 3. **Monitoramento da Saúde do Modelo**
   - **MLflow**: Possui funcionalidades de model registry e tracking, permitindo monitorar a degradação do modelo ao longo do tempo.
   - **Streamlit**: Pode ser utilizado para criar painéis interativos e visualizar métricas do modelo em tempo real.

### 4. **Atualização de Modelo**
   - **MLflow**: Possui um registry para gerenciar diferentes versões dos modelos, facilitando a atualização e rollback de versões anteriores.
   - **PyCaret**: Implementa pipelines automatizados que incluem reentrenamento de modelos com novos dados.

### 5. **Provisionamento (Deployment)**
   - **MLflow**: Oferece integração com APIs REST para servir modelos como microserviços.
   - **Streamlit**: Permite construir interfaces interativas para testar previsões do modelo de forma simples e rápida.

---

### **Impacto da Escolha de Treino e Teste no Modelo Final**
A forma como os dados de treino e teste são divididos pode influenciar diretamente o desempenho e a capacidade de generalização do modelo. Algumas considerações importantes incluem:

1. **Representatividade dos Dados**  
   - Se os dados de treino não forem representativos da distribuição real dos dados, o modelo pode não aprender padrões generalizáveis.
   - Se o conjunto de teste for muito diferente do de treino, o modelo pode ter um desempenho artificialmente ruim.

2. **Overfitting e Underfitting**  
   - **Overfitting:** O modelo pode memorizar padrões específicos do conjunto de treino e ter um desempenho ruim em novos dados (teste).
   - **Underfitting:** Se o conjunto de treino for muito pequeno ou não representativo, o modelo pode não aprender padrões relevantes.

3. **Estratificação e Balanceamento**  
   - Se o conjunto de treino e teste não mantiver a mesma proporção das classes (no caso de classificação), o modelo pode aprender viéses indesejados.

---

## Questão 4

Com base no diagrama gerado, que ilustra um projeto usando **Kedro**, podemos identificar diversos artefatos criados ao longo do pipeline de dados. 
Abaixo está a lista dos principais **artefatos**, com uma descrição detalhada da composição de cada um:

---

### 1. **dataset_kobe** (Catálogo: `raw`)
- **Tipo**: Fonte de dados bruta.
- **Composição**:
  - Arquivo CSV, Excel, JSON, banco de dados ou API contendo dados não tratados.
  - Inclui todas as variáveis originais, ruídos e possíveis inconsistências.
  - Cadastrado no `catalog.yml` com tipo `raw`.

---

### 2. **data_filtered**
- **Tipo**: Dados pré-processados.
- **Composição**:
  - Resultado da limpeza e filtragem realizada pelo node `PreparacaoDados`.
  - Pode incluir:
    - Remoção de outliers ou nulos.
    - Conversão de tipos de dados.
    - Aplicação de filtros lógicos (ex: apenas jogadas feitas em partidas oficiais).
  - Cadastrado no `catalog.yml` como um dataset intermediário.

---

### 3. **base_train / base_test** (Catálogo: `entrada`)
- **Tipo**: Dados de treino e teste.
- **Composição**:
  - Conjunto de dados particionado a partir de `data_filtered`.
  - Pode incluir engenharia de atributos, normalização, codificação categórica etc.
  - Usado para alimentar o node de `Treinamento`.
  - Armazenado separadamente em arquivos como `base_train.parquet` e `base_test.parquet`.

---

### 4. **Modelo Treinado** (registrado no MLflow)
- **Tipo**: Modelo de machine learning.
- **Composição**:
  - Objeto serializado do modelo (ex: `sklearn`, `xgboost`, etc.).
  - Metainformações como:
    - Hiperparâmetros usados.
    - Métricas de performance (ex: acurácia, F1-score).
    - Artefatos complementares como scaler, encoder, etc.
  - Registrado no **MLflow Model Registry** com versionamento.

---

### 5. **Registro no MLflow**
- **Tipo**: Metadata tracking.
- **Composição**:
  - Logs de execução do experimento.
  - Parâmetros de input (hiperparâmetros).
  - Métricas de avaliação.
  - Arquivos de saída como o modelo `.pkl`, gráficos, etc.
  - Interface visual e API de consulta.

---

### 6. **Aplicação Streamlit**
- **Tipo**: Interface web.
- **Composição**:
  - Código Python com lógica de front-end interativo.
  - Elementos como:
    - Campos de input do usuário.
    - Lógica de requisição ao modelo via MLflow (ou diretamente).
    - Exibição de outputs como previsão, gráficos, explicações.
  - Conectada ao modelo treinado para inferência em tempo real.

---

### 7. **Nodes (Funções do pipeline)**
Embora não sejam arquivos em si, os *nodes* são artefatos de código fundamentais:
- **PreparacaoDados**:
  - Função responsável por ingestão e limpeza dos dados brutos.
- **Treinamento**:
  - Função que recebe os dados tratados e executa o treinamento do modelo.

---
## Questão 5

### **Implementação e Execução do Pipeline "PreparacaoDados"**
Todo o pipeline foi integrado com MLFlow para registro e acompanhamento dos experimentos.
<img src="docs\images\pipeline-run-preparacao-log.png">

Ao executar o pipeline, as seguintes métricas foram geradas no MLflow:
<img src="docs\images\mlflow-preparacao.png">

E também os arquivos gerados:
<img src="docs\images\pipeline-preparacao-artefatos.png">

### **Estratégias para Minimizar o Viés de Dados**
1. **Divisão Estratificada**  
   - Para problemas de classificação, a técnica **stratified split** garante que a distribuição da variável alvo seja semelhante nos conjuntos de treino e teste.  
   - Isso evita que o modelo aprenda padrões enviesados por distribuições desbalanceadas.

   **Exemplo no Scikit-Learn:**  
   ```python
   from sklearn.model_selection import train_test_split

   train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["shot_made_flag"], random_state=42)
   ```

2. **Aumento de Dados (Data Augmentation)**  
   - Quando há poucas amostras de certas classes, pode-se gerar novos dados sintéticos para equilibrar as classes.
   - Técnicas como **SMOTE (Synthetic Minority Over-sampling Technique)** podem ser usadas para balancear classes desbalanceadas.

3. **Cross-Validation (Validação Cruzada)**  
   - Em vez de usar uma única divisão treino/teste, a validação cruzada (ex.: k-fold cross-validation) permite testar o modelo em diferentes divisões dos dados, reduzindo o risco de viés na amostragem.

   **Exemplo de K-Fold Cross-Validation no Scikit-Learn:**  
   ```python
   from sklearn.model_selection import KFold, cross_val_score

   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
   print(f"Média das acurácias: {scores.mean()}")
   ```

4. **Remoção de Dados Redundantes e Limpeza**  
   - Identificar e remover duplicatas ou dados inconsistentes pode evitar que o modelo aprenda padrões irrelevantes.

5. **Testar com Dados de Produção**  
   - Se possível, testar o modelo com dados reais que ele encontrará na produção para garantir que os resultados sejam confiáveis.

## Questão 6

O pipeline foi deviamente registrado no MLflow:
<img src="docs\images\mlflow-treinamento.png">

Durante a execução do pipeline, o seguinte output no console foi gerado:

---

#### **Árvore de Decisão:**
<img src="docs\images\training-metrics-dt.png">

---

#### **Regeressão Logística:**
<img src="docs\images\training-metrics-lr.png">

---
Com base nas métricas de validação cruzada (10 folds) apresentadas nas imagens dos dois modelos — **Regressão Logística** e **Árvore de Decisão** — o modelo mais adequado para finalização é:

### ✅ **Modelo Escolhido: Regressão Logística**

### 📊 **Comparativo das principais métricas (média dos folds)**

| Métrica     | Regressão Logística | Árvore de Decisão |
|-------------|----------------------|--------------------|
| **Accuracy** | **0.5781**           | 0.5264             |
| **AUC**      | **0.6085**           | 0.5160             |
| **Recall**   | 0.4921               | **0.5530**         |
| **Precision**| **0.5669**           | 0.5034             |
| **F1 Score** | **0.5267**           | 0.5142             |
| **Kappa**    | **0.1496**           | 0.0657             |
| **MCC**      | **0.1509**           | 0.0662             |

---

### 🧠 **Justificativa da escolha**

1. **Acurácia Geral Superior**: A regressão logística alcançou uma média de acurácia 5 pontos percentuais acima da árvore de decisão (57.8% vs. 52.6%).

2. **Melhor Separação entre Classes (AUC)**: O AUC da regressão logística (0.6085) indica maior capacidade de distinguir as classes corretamente. O modelo de árvore tem AUC próximo de 0.5, o que sugere desempenho similar ao aleatório.

3. **Métricas de Balanceamento (Kappa e MCC)**: Ambas são substancialmente mais altas na regressão logística, o que reforça que o modelo está aprendendo padrões úteis, e não apenas se ajustando ao desbalanceamento ou aleatoriedade.

4. **Recall ligeiramente inferior, mas compensado**: Embora a árvore de decisão tenha maior *recall* (sensibilidade), ela perde em todas as outras métricas, o que torna o modelo menos robusto como um todo.
