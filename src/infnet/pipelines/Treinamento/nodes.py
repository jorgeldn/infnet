import mlflow
import mlflow.sklearn
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import log_loss, f1_score, accuracy_score
from mlflow.models.signature import infer_signature

def train_logistic_model(base_train, base_test):
    """
    Treina um modelo de Regressão Logística e registra métricas, assinatura e input_example no MLflow.
    """
    X_test = base_test.drop(columns=["shot_made_flag"])
    y_test = base_test["shot_made_flag"]

    with mlflow.start_run(run_name="Regressao_Logistica", nested=True):
        exp = ClassificationExperiment()
        exp.setup(data=base_train, target="shot_made_flag", session_id=42, log_experiment=False)
        logistic_model = exp.create_model("lr")

        preds = exp.predict_model(logistic_model, data=X_test)
        y_pred_prob = preds["prediction_score"]
        y_pred_label = preds["prediction_label"]

        # Registro de métricas
        mlflow.log_param("Modelo", "Regressão Logística")
        mlflow.log_metric("Log Loss", log_loss(y_test, y_pred_prob))
        mlflow.log_metric("F1 Score", f1_score(y_test, y_pred_label))
        mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred_label))

        # Assinatura e input_example
        signature = infer_signature(X_test, logistic_model.predict_proba(X_test))
        input_example = X_test.head(3)

        # Registro do modelo com schema
        mlflow.sklearn.log_model(
            sk_model=logistic_model,
            artifact_path="Modelo_Logistico",
            signature=signature,
            input_example=input_example,
        )

        return logistic_model


def train_decision_tree_model(base_train, base_test):
    """
    Treina um modelo de Árvore de Decisão e registra métricas, assinatura e input_example no MLflow.
    """
    X_test = base_test.drop(columns=["shot_made_flag"])
    y_test = base_test["shot_made_flag"]

    with mlflow.start_run(run_name="Arvore_Decisao", nested=True):
        exp = ClassificationExperiment()
        exp.setup(data=base_train, target="shot_made_flag", session_id=42, log_experiment=False)
        decision_tree_model = exp.create_model("dt")

        preds = exp.predict_model(decision_tree_model, data=X_test)
        y_pred_prob = preds["prediction_score"]
        y_pred_label = preds["prediction_label"]

        # Registro de métricas
        mlflow.log_param("Modelo", "Árvore de Decisão")
        mlflow.log_metric("Log Loss", log_loss(y_test, y_pred_prob))
        mlflow.log_metric("F1 Score", f1_score(y_test, y_pred_label))
        mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred_label))

        # Assinatura e input_example
        signature = infer_signature(X_test, decision_tree_model.predict_proba(X_test))
        input_example = X_test.head(3)

        # Registro do modelo com schema
        mlflow.sklearn.log_model(
            sk_model=decision_tree_model,
            artifact_path="Modelo_Arvore",
            signature=signature,
            input_example=input_example,
        )

        return decision_tree_model
