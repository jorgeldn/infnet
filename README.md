<img src="docs\images\infnet-logo.png" width="200">

# Projeto da Disciplina de Engenharia de Machine Learning

---
## Aluno: Jorge Luiz do Nascimento Júnior
Git do projeto:
[https://github.com/jorgeldn/infnet](https://github.com/jorgeldn/infnet)

## 📜 Overview

Desenvolver um preditor de arremessos usando duas abordagens (regressão e classificação) para prever se o "Black Mamba" (apelido de Kobe) acertou ou errou a cesta.
Na pasta `data/01_raw` estão disposiveis os arquivos: **dataset_kobe_dev.parquet** e **dataset_kobe_prod.parquet** alvos de estudo deste projeto.

## 🌐 Configuração do ambiente de desenvolvimento

Para desenvolver e executar o projeto de disciplina, foi instalado o `python 3.11` e criado um ambiente virtual utilizando o **VENV** com o seguinte comando:

```bash
python -m venv venv-pos-ia
```

Foi gerado um arquivo `requirements.in` com as dependências do projeto.
O arquivo foi compilado com o comando:

```bash
pip-compile requirements.in
``` 

gerando o arquivo `requirements.txt`.

Para instalação de todas as bibliotecas, executar:

```bash
pip install -r requirements.txt
```

## 🎯Questão 1: Estrutura e solução do projeto?

---
Este projeto foi desenvolvido utilizando o framework Kedro versão `kedro 0.19.11`.

O projeto segue a seguinte estrutura:
```
📂 .  
├── 📂 conf  
│   ├── 📂 base  
│   │   ├── 📄 catalog.yml  
│   │   ├── 📄 parameters.yml 
├── 📂 data  
│   ├── 📂 01_raw  
│   │   ├── 📄 dataset_kobe_dev.parquet  
│   │   ├── 📄 dataset_kobe_prod.parquet  
│   ├── 📂 02_intermediate  
│   │   ├── 📄 base_test.parquet  
│   │   ├── 📄 base_train.parquet  
│   │   ├── 📄 data_filtered.parquet  
│   ├── 📂 03_primary  
│   ├── 📂 04_feature  
│   ├── 📂 05_model_input  
│   ├── 📂 06_models  
│   ├── 📂 07_model_output  
│   │   ├── 📄 resultados_aplicacao.parquet  
│   ├── 📂 08_reporting  
├── 📂 docs  
│   ├── 📂 images
├── 📂 mlruns  
├── 📂 notebooks  
├── 📂 src  
│   ├── 📂 infnet  
│   │   ├── 📂 pipelines  
│   │   │   ├── 📂 Aplicacao  
│   │   │   │   ├── 📄 __init__.py  
│   │   │   │   ├── 📄 nodes.py  
│   │   │   │   ├── 📄 pipeline.py  
│   │   │   ├── 📂 PreparacaoDados  
│   │   │   │   ├── 📄 __init__.py  
│   │   │   │   ├── 📄 nodes.py  
│   │   │   │   ├── 📄 pipeline.py  
│   │   │   ├── 📂 Treinamento  
│   │   │   │   ├── 📄 __init__.py  
│   │   │   │   ├── 📄 nodes.py  
│   │   │   │   ├── 📄 pipeline.py   
│   │   ├── 📄 sklearn_proba_wrapper.py  
├── 📂 streamlit  
│   ├── 📄 app.py  
├── 📂 tests   
├── 📄 pyproject.toml  
├── 📄 README.md  
├── 📄 requirements.in  
├── 📄 requirements.txt
```
## 🎯Questão 2: Diagrama de etapas do projeto

---
<img src="docs\images\diagram.png">

## 🎯Questão 3: Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines?

---
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


### 📌**Impacto da Escolha de Treino e Teste no Modelo Final**
A forma como os dados de treino e teste são divididos pode influenciar diretamente o desempenho e a capacidade de generalização do modelo. Algumas considerações importantes incluem:

1. **Representatividade dos Dados**  
   - Se os dados de treino não forem representativos da distribuição real dos dados, o modelo pode não aprender padrões generalizáveis.
   - Se o conjunto de teste for muito diferente do de treino, o modelo pode ter um desempenho artificialmente ruim.

2. **Overfitting e Underfitting**  
   - **Overfitting:** O modelo pode memorizar padrões específicos do conjunto de treino e ter um desempenho ruim em novos dados (teste).
   - **Underfitting:** Se o conjunto de treino for muito pequeno ou não representativo, o modelo pode não aprender padrões relevantes.

3. **Estratificação e Balanceamento**  
   - Se o conjunto de treino e teste não mantiver a mesma proporção das classes (no caso de classificação), o modelo pode aprender viéses indesejados.

## 🎯Questão 4

---

Com base no diagrama gerado, que ilustra um projeto usando **Kedro**, podemos identificar diversos artefatos criados ao longo do pipeline de dados. 
Abaixo está a lista dos principais **artefatos**, com uma descrição detalhada da composição de cada um:

### 1. 📑 **dataset_kobe** (Catálogo: `raw`)
- **Tipo**: Fonte de dados bruta.
- **Composição**:
  - Arquivo Parquet.
  - Inclui todas as variáveis originais, ruídos e possíveis inconsistências.
  - Cadastrado no `catalog.yml` com tipo `raw`.

#### Estrutura da Tabela

| #  | Coluna               | Tipo     | Descrição |
|----|----------------------|----------|-----------|
| 0  | action_type         | object   | Tipo específico da ação de arremesso (ex: Jump Shot, Layup, Dunk). |
| 1  | combined_shot_type  | object   | Tipo geral de arremesso (ex: Jump Shot, Bank Shot, Dunk). |
| 2  | game_event_id       | int64    | Identificação única do evento dentro da partida. |
| 3  | game_id             | int64    | Identificação única da partida. |
| 4  | lat                 | float64  | Latitude da posição do arremesso na quadra. |
| 5  | loc_x               | int64    | Coordenada X do local do arremesso em relação à quadra. |
| 6  | loc_y               | int64    | Coordenada Y do local do arremesso em relação à quadra. |
| 7  | lon                 | float64  | Longitude da posição do arremesso na quadra. |
| 8  | minutes_remaining   | int64    | Minutos restantes no quarto em que o arremesso foi realizado. |
| 9  | period              | int64    | Número do período da partida (ex: 1, 2, 3, 4, OT). |
| 10 | playoffs           | int64    | Indica se a partida foi nos playoffs (1) ou na temporada regular (0). |
| 11 | season             | object   | Temporada do jogo no formato "YYYY-YY" (ex: "2001-02"). |
| 12 | seconds_remaining  | int64    | Segundos restantes no quarto em que o arremesso foi realizado. |
| 13 | shot_distance      | int64    | Distância do arremesso em pés. |
| 14 | shot_made_flag     | float64  | Indica se o arremesso foi convertido (1) ou não (0). |
| 15 | shot_type          | object   | Tipo de arremesso baseado no número de pontos (2PT Field Goal ou 3PT Field Goal). |
| 16 | shot_zone_area     | object   | Área da quadra onde o arremesso ocorreu (ex: Right Side, Left Side, Center). |
| 17 | shot_zone_basic    | object   | Classificação do tipo de zona do arremesso (ex: Restricted Area, Mid-Range, Backcourt). |
| 18 | shot_zone_range    | object   | Distância do arremesso (ex: 8-16 ft., 16-24 ft., 24+ ft.). |
| 19 | team_id            | int64    | Identificação única do time de Kobe Bryant (Los Angeles Lakers). |
| 20 | team_name          | object   | Nome do time (Los Angeles Lakers). |
| 21 | game_date          | object   | Data da partida no formato YYYYMMDD. |
| 22 | matchup           | object   | Identificação do confronto (ex: LAL @ BOS, LAL vs MIA). |
| 23 | opponent          | object   | Nome do time adversário. |
| 24 | shot_id           | int64    | Identificação única do arremesso. |

##### Observações
- O dataset contém 24.271 registros.
- A coluna `shot_made_flag` possui valores nulos, indicando arremessos cuja conversão não foi informada.

---

### 2. 📑 **data_filtered** (Catálogo: `entrada`)
- **Tipo**: Dados pré-processados.
- **Composição**:
  - Resultado da limpeza e filtragem realizada pelo pipeline `PreparacaoDados` no node `prepare_data`.
  - As transformações incluem:
    - Remoção de outliers ou nulos.
    - Conversão de tipos de dados.
  - Cadastrado no `catalog.yml` como um dataset intermediário.

##### Somente as colunas foram selecionadas para o `data_filtered`:

| Coluna               | Tipo     | Descrição |
|----------------------|----------|-----------|
| lat                 | float64  | Latitude da posição do arremesso na quadra. |
| lon                 | float64  | Longitude da posição do arremesso na quadra. |
| minutes_remaining   | int64    | Minutos restantes no quarto em que o arremesso foi realizado. |
| period              | int64    | Número do período da partida (ex: 1, 2, 3, 4, OT). |
| playoffs           | int64    | Indica se a partida foi nos playoffs (1) ou na temporada regular (0). |
| shot_distance      | int64    | Distância do arremesso em pés. |


### 3. 📑 **base_train / base_test** (Catálogo: `entrada`)
- **Tipo**: Dados de treino e teste.
- **Composição**:
  - Conjunto de dados particionado a partir de `data_filtered`.
  - Usado para alimentar o node de `Treinamento`.
  - Armazenado separadamente em arquivos como `base_train.parquet` e `base_test.parquet`.


### 4. 🧮 **Modelo Treinado** (registrado no MLflow)
- **Tipo**: Modelo de machine learning.
- **Composição**:
  - Objeto serializado dos modelos `Logistic Regression` e `Decision Tree Classifier`.
  - Metainformações como:
    - Hiperparâmetros usados.
    - Métricas de performance (ex: acurácia, F1-score).
    - Artefatos complementares como scaler, encoder, etc.
  - Registrado no **MLflow Model Registry** com versionamento.


### 5. 📈 **Registro no MLflow**
- **Tipo**: Metadata tracking.
- **Composição**:
  - Logs de execução do experimento.
  - Parâmetros de input (hiperparâmetros).
  - Métricas de avaliação.
  - Arquivos de saída como o modelo `.pkl`, gráficos, etc.
  - Interface visual e API de consulta.


### 6. 📟 **Aplicação Streamlit**
- **Tipo**: Interface web.
- **Composição**:
  - Código Python com lógica de front-end interativo.
  - Elementos como:
    - Campos de input do usuário.
    - Lógica de requisição ao modelo via MLflow (ou diretamente).
    - Exibição de outputs com probabilidades.
  - Conectada ao modelo treinado para inferência em tempo real.


### 7. ⚙️ **Nodes (Funções do pipeline)**
Embora não sejam arquivos em si, os *nodes* são artefatos de código fundamentais:
- **PreparacaoDados**:
  - Função responsável por ingestão e limpeza dos dados brutos.
- **Treinamento**:
  - Funções que recebem os dados tratados e executa o treinamento dos modelos.


## 🎯Questão 5

---
No prompt de comando, executado o seguinte comando:
```bash
kedro run --pipeline=PreparacaoDados
```

ℹ️ Sobre o dataset `data_filtered` 
> Dimensão do dataset após tratamento: **_20285 linhas e 7 colunas_**.

### **Implementação e Execução do Pipeline "PreparacaoDados"**
Todo o pipeline foi integrado com MLFlow para registro e acompanhamento dos experimentos.
<img src="docs\images\pipeline-run-preparacao-log.png">

Ao executar o pipeline, as seguintes métricas foram geradas no MLflow:
<img src="docs\images\mlflow-preparacao.png">

E também os arquivos gerados:
<img src="docs\images\pipeline-preparacao-artefatos.png">

### ✅ **Estratégias para Minimizar o Viés de Dados**
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

## 🎯Questão 6

---
No prompt de comando, executado o seguinte comando:
```bash
kedro run --pipeline=Treinamento
```

O pipeline foi deviamente registrado no MLflow:
<img src="docs\images\mlflow-treinamento.png">

Durante a execução do pipeline, o seguinte output no console foi gerado:

#### ✅ **Árvore de Decisão:**
<img src="docs\images\training-metrics-dt.png">

---

#### ✅**Regeressão Logística:**
<img src="docs\images\training-metrics-lr.png">

---
Com base nas métricas de validação cruzada (10 folds) apresentadas nas imagens dos dois modelos — **Regressão Logística** e **Árvore de Decisão** — o modelo mais adequado para finalização é:

### ✅ **Modelo Escolhido: Regressão Logística**

##### **Comparativo das principais métricas (média dos folds)**

| Métrica     | Regressão Logística | Árvore de Decisão |
|-------------|----------------------|--------------------|
| **Accuracy** | **0.5781**           | 0.5264             |
| **AUC**      | **0.6085**           | 0.5160             |
| **Recall**   | 0.4921               | **0.5530**         |
| **Precision**| **0.5669**           | 0.5034             |
| **F1 Score** | **0.5267**           | 0.5142             |
| **Kappa**    | **0.1496**           | 0.0657             |
| **MCC**      | **0.1509**           | 0.0662             |


### 🧠 **Justificativa da escolha**

1. **Acurácia Geral Superior**: A regressão logística alcançou uma média de acurácia 5 pontos percentuais acima da árvore de decisão (57.8% vs. 52.6%).

2. **Melhor Separação entre Classes (AUC)**: O AUC da regressão logística (0.6085) indica maior capacidade de distinguir as classes corretamente. O modelo de árvore tem AUC próximo de 0.5, o que sugere desempenho similar ao aleatório.

3. **Métricas de Balanceamento (Kappa e MCC)**: Ambas são substancialmente mais altas na regressão logística, o que reforça que o modelo está aprendendo padrões úteis, e não apenas se ajustando ao desbalanceamento ou aleatoriedade.

4. **Recall ligeiramente inferior, mas compensado**: Embora a árvore de decisão tenha maior *recall* (sensibilidade), ela perde em todas as outras métricas, o que torna o modelo menos robusto como um todo.

---

## 🎯Questão 7

Após desenvolver e executar o pipeline, o seguinte erro foi gerado no console:
<img src="docs\images\pipeline-aplicacao-erro.png">

Significa que os tipos de dados do DataFrame de produção não é aderente ao de desenvolvimento. Isso ocorreu pois os tipos de dados passados para o modelo via predict não bateram com a assinatura registrada do modelo.

Para contornar o problema de falha na execução, foi necessário editar o código para corrigir os tipos de dados.
<img src="docs\images\mlflow-aplicacao-plot.png">

### 📌 **O que mudou entre a base de treino (`kobe_dev`) e a base de produção (`kobe_prod`)?**

Pelos indícios observados:

| Aspecto | Base de Treinamento (`dev`) | Base de Produção (`prod`) |
|--------|------------------------------|----------------------------|
| **Colunas** | 7 (features + target) | 24 (colunas completas do dataset original) |
| **Filtragem** | Pré-processada, com somente variáveis relevantes e sem nulos | Crua, sem filtragem, com valores faltantes no `shot_made_flag` |
| **Tipos de dados** | Todos como `float64`, compatíveis com MLflow | Muitos como `int64`, `object`, e com `NaN`s |

#### 🧠 Conclusão:
A base de produção **não foi tratada com o mesmo pré-processamento** que a base de treino, o que pode causar inconsistência nas previsões e quebra de performance.

#### 🧠 Justificativa

O modelo **não é diretamente aderente** à base de produção por três motivos principais:

1. A produção não passou por pré-processamento (tem campos extras, tipos errados e valores nulos).
2. As variáveis relevantes estão misturadas com outras não usadas pelo modelo.
3. A presença de `NaN` no target impedia a avaliação direta do desempenho.


### ✅ **Monitoramento da saúde do modelo**

A saúde de um modelo pode ser monitorada em duas frentes principais:

- **Performance**: como o modelo está se saindo?
- **Dados**: os dados que entram no modelo continuam parecidos com os que ele foi treinado?


### **1. Quando a variável resposta (target) está disponível em produção**

Este é o **melhor cenário possível**, pois permite **medir o desempenho real** do modelo com dados atualizados.

#### 🎯 O que pode ser monitorado:
- **Métricas de performance** como:
  - `log_loss`, `f1_score`, `accuracy`, `precision`, `recall`
- **Atrasos entre previsão e rótulo real** (tempo de feedback)
- **Drift de performance**: comparar as métricas com benchmarks anteriores

#### 🛠️ Ferramentas e abordagens:
- Executar o **pipeline de aplicação** com o target incluído
- Registrar as métricas no MLflow ou ferramentas como Evidently, Prometheus, etc.
- Programar **dashboards** em Streamlit, Superset, Grafana ou Power BI

---

### **2. Quando a variável resposta NÃO está disponível em produção**

Neste caso, você **não pode medir diretamente a performance**. Mas ainda é possível monitorar **a integridade e a aderência dos dados**.

#### 🔎 O que monitorar:

##### a) **Data Drift (mudança nos dados de entrada)**
- Mudança na distribuição das variáveis
- Novos valores em variáveis categóricas
- Mudança de média, mediana, desvio padrão

##### b) **Feature Importance Drift**
- Ver se a importância das variáveis mudou muito com o tempo

##### c) **Score Stability**
- A distribuição das probabilidades do modelo ao longo do tempo
- Ex: média da probabilidade de classe 1

#### 🛠️ Ferramentas e abordagens:
- Usar **Evidently** (Python) para:
  - Data Drift Report
  - Targetless Monitoring
- **Alerts automáticos** se alguma feature sair do intervalo de confiança
- Logar a entrada do modelo com timestamp para auditoria futura


#### 📊 Exemplo prático de métricas monitoráveis sem `y`:

| Métrica                 | Com `y` | Sem `y` |
|------------------------|---------|---------|
| Accuracy / F1 Score    | ✅      | ❌      |
| Log Loss               | ✅      | ❌      |
| Probabilidade média    | ✅/❌   | ✅      |
| Mudança de distribuição| ✅/❌   | ✅      |
| Tempo de predição      | ✅      | ✅      |
| Taxa de erros técnicos | ✅      | ✅      |


### 🧩 Conclusão

| Situação                     | O que monitorar                                |
|-----------------------------|------------------------------------------------|
| **Com target disponível**   | Métricas de performance + integridade de dados |
| **Sem target disponível**   | Drift de dados + distribuição de scores        |

Ambos os cenários exigem ações automáticas como **logs, alertas e auditorias**, e ferramentas como **MLflow, Evidently, Prometheus e Streamlit** ajudam nesse processo.

---
Quando colocamos um modelo em produção, é fundamental definir **estratégias de retreinamento** para garantir que ele continue relevante, preciso e confiável com o passar do tempo.

Essas estratégias podem ser divididas em **duas abordagens principais**:

### 🔁 **1. Estratégia Reativa de Retreinamento**

> O modelo só é retreinado **quando há sinais claros de degradação**.

#### 📌 Características:
- Baseada em **monitoramento constante** da performance
- Retreinamento **acontece após** o modelo apresentar queda nas métricas
- Depende da **disponibilidade da variável resposta (target)**

#### 🧠 Exemplos de gatilhos:
- F1 Score ou Accuracy caiu abaixo de um limite (ex: 5% abaixo do benchmark)
- Log Loss subiu acima de um limiar
- Aumento de reclamações ou erros operacionais
- Atraso no tempo de resposta devido a complexidade crescente

#### ✅ Vantagens:
- Evita retrabalho desnecessário
- É eficiente quando há **feedback contínuo** do target

#### ❌ Desvantagens:
- É **reativa**: o modelo já degradou quando o retreinamento começa
- Pode gerar **impactos negativos** antes da correção (ex: perda de vendas, decisões ruins)


### 📅 **2. Estratégia Preditiva (ou Proativa) de Retreinamento**

> O modelo é retreinado **em ciclos planejados ou com base em sinais antecipados**, mesmo sem queda visível de performance.

#### 📌 Características:
- Foco em **prevenir** problemas futuros
- Usa análise de **data drift** ou **mudança de contexto**
- Pode ser baseada em **agendamento (ex: mensal, trimestral)**

#### 🧠 Exemplos de gatilhos:
- Mudança estatística significativa nas variáveis de entrada (data drift)
- Aumento na frequência de valores ausentes, anomalias ou categorias novas
- Modelo em produção atingiu um número X de novas amostras
- **Janela de tempo definida** (retraining a cada 60 dias, por exemplo)

#### ✅ Vantagens:
- Mantém o modelo sempre fresco e adaptado
- Reduz risco de deterioração súbita

#### ❌ Desvantagens:
- Pode ser **custoso** (mais consumo computacional e tempo de validação)
- Risco de **overfitting ao novo contexto** se não for bem gerenciado

---

#### 🧩 Comparativo das Estratégias

| Aspecto                    | Estratégia Reativa        | Estratégia Preditiva         |
|---------------------------|---------------------------|------------------------------|
| Base                      | Métricas de performance   | Tempo, volume ou sinais de drift |
| Exige `target`?           | Sim                       | Não necessariamente          |
| Ação                      | Após problema aparecer    | Antes do problema acontecer  |
| Custo computacional       | Mais baixo                | Mais alto                    |
| Risco de impacto negativo | Maior                     | Menor                        |

---

### 📟 Streamlit App

Foi implementada uma aplicação Streamlit para visualização dos dados e monitoramento do modelo.
O App possui 2 abas:

- **Aba 1**: Previsão --> (Consome a API do MLflow)
- **Aba 2**: Monitoramento da Operação

> Para executar a app, basta rodar o arquivo `streamlit/app.py` com o comando: 
```bash
streamlit run app.py
```

<img src="docs\images\streamlit-aba-prev.png">
<img src="docs\images\streamlit-aba-mon-01.png">
<img src="docs\images\streamlit-aba-mon-02.png">
<img src="docs\images\streamlit-aba-mon-03.png">