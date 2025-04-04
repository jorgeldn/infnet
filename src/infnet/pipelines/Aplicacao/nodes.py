import mlflow
import pandas as pd
from sklearn.metrics import log_loss, f1_score
import mlflow.pyfunc


def aplicar_modelo_producao(dataset_kobe_prod):
    """
    Aplica o modelo logistic_regression do MLflow ao conjunto de dados de produção.
    Retorna o dataframe com previsões e registra métricas no MLflow.
    """
    # Elimina registros onde shot_made_flag está nulo
    dataset_kobe_prod = dataset_kobe_prod.dropna(subset=["shot_made_flag"])

    # Separar features e target
    X_prod = dataset_kobe_prod.drop(columns=["shot_made_flag"])

    # Corrigir os tipos para casar com a signature do modelo
    X_prod = X_prod.astype({
        "lat": float,
        "lon": float,
        "minutes_remaining": float,
        "period": float,
        "playoffs": float,
        "shot_distance": float
    })
    y_true = dataset_kobe_prod["shot_made_flag"]

    # Carregar modelo mais recente
    model_uri = "models:/kobe_lr_model/2"
    model = mlflow.pyfunc.load_model(model_uri)

    # Iniciar run principal
    with mlflow.start_run(run_name="PipelineAplicacao", nested=True):
        # Realizar previsõescls
        y_pred_prob = model.predict(X_prod)  # retorna predict_proba
        y_pred_label = (y_pred_prob[:, 1] >= 0.5).astype(int)

        # Calcular métricas
        loss = log_loss(y_true, y_pred_prob)
        f1 = f1_score(y_true, y_pred_label)

        # Log de métricas
        mlflow.log_metric("log_loss_aplicacao", loss)
        mlflow.log_metric("f1_score_aplicacao", f1)

        # Construir dataframe final com outputs
        resultados = X_prod.copy()
        resultados["y_true"] = y_true
        resultados["prob_0"] = y_pred_prob[:, 0]
        resultados["prob_1"] = y_pred_prob[:, 1]
        resultados["y_pred"] = y_pred_label

        # Salvar como artefato no Data Catalog (output do node)
        return resultados
