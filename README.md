# Projeto da Disciplina de Engenharia de Machine Learning

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Este projeto foi desenvolvido utilizando o framework Kedro versão `kedro 0.19.11`.

## Dependencias

Todas as dependencias estão listadas no `requirements.txt`.

Para instalação, executar:

```
pip install pip-tools
```

```
pip-compile requirements.in
```

```
pip install -r requirements.txt
```
## Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines?

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