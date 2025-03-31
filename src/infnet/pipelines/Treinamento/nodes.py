import mlflow
import mlflow.sklearn
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import log_loss, f1_score


def train_models(train_data, test_data):
    """
    Treina modelos de Regressão Logística e Árvore de Decisão usando PyCaret,
    registrando métricas no MLflow.
    """

    X_test = test_data.drop(columns=["shot_made_flag"])
    y_test = test_data["shot_made_flag"]

    with mlflow.start_run(run_name="Treinamento", nested=True):
        exp = ClassificationExperiment()

        # Treinar Regressão Logística
        with mlflow.start_run(run_name="Treinamento_Regressao_Logistica", nested=True):
            exp.setup(data=train_data, target="shot_made_flag", session_id=42, log_experiment=False)
            logistic_model = exp.create_model("lr")  # 'lr' -> Logistic Regression

            # Fazer previsões e calcular log loss
            y_pred_prob = exp.predict_model(logistic_model, data=X_test)["prediction_score"]
            log_loss_lr = log_loss(y_test, y_pred_prob)

            # Log no MLflow
            mlflow.log_param("Modelo", "Regressão Logística")
            mlflow.log_metric("Log Loss", log_loss_lr)

            # Salvar o modelo no MLflow
            mlflow.sklearn.log_model(logistic_model, "Modelo_Logistico")

        # Treinar Árvore de Decisão
        with mlflow.start_run(run_name="Treinamento_Arvore_Decisao", nested=True):
            decision_tree_model = exp.create_model("dt")  # 'dt' -> Decision Tree

            # Fazer previsões e calcular métricas
            y_pred_prob_dt = exp.predict_model(decision_tree_model, data=X_test)["prediction_score"]
            y_pred_dt = exp.predict_model(decision_tree_model, data=X_test)["prediction_label"]
            log_loss_dt = log_loss(y_test, y_pred_prob_dt)
            f1_dt = f1_score(y_test, y_pred_dt)

            # Log no MLflow
            mlflow.log_param("Modelo", "Árvore de Decisão")
            mlflow.log_metric("Log Loss", log_loss_dt)
            mlflow.log_metric("F1 Score", f1_dt)

            # Salvar o modelo no MLflow
            mlflow.sklearn.log_model(decision_tree_model, "Modelo_Arvore")




