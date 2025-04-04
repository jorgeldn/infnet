import mlflow
import mlflow.pyfunc
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import log_loss, f1_score, accuracy_score
from mlflow.models.signature import infer_signature
import joblib
import tempfile
import os
from infnet.sklearn_proba_wrapper import SklearnProbaWrapper


def train_logistic_model(base_train, base_test):
    X_test = base_test.drop(columns=["shot_made_flag"])
    y_test = base_test["shot_made_flag"]

    with mlflow.start_run(run_name="Regressao_Logistica", nested=True):
        exp = ClassificationExperiment()
        exp.setup(data=base_train, target="shot_made_flag", session_id=42, log_experiment=False)
        logistic_model = exp.create_model("lr")

        preds = exp.predict_model(logistic_model, data=X_test)
        y_pred_prob = preds["prediction_score"]
        y_pred_label = preds["prediction_label"]

        mlflow.log_param("Modelo", "Regressão Logística")
        mlflow.log_metric("Log Loss", log_loss(y_test, y_pred_prob))
        mlflow.log_metric("F1 Score", f1_score(y_test, y_pred_label))
        mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred_label))

        # Salvar modelo local e registrar com wrapper
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pkl")
            joblib.dump(logistic_model, model_path)

            signature = infer_signature(X_test, logistic_model.predict_proba(X_test))
            input_example = X_test.head(3)

            mlflow.pyfunc.log_model(
                artifact_path="Modelo_Logistico",
                python_model=SklearnProbaWrapper(),
                artifacts={"model_path": model_path},
                signature=signature,
                input_example=input_example,
            )

        return logistic_model


def train_decision_tree_model(base_train, base_test):
    X_test = base_test.drop(columns=["shot_made_flag"])
    y_test = base_test["shot_made_flag"]

    with mlflow.start_run(run_name="Arvore_Decisao", nested=True):
        exp = ClassificationExperiment()
        exp.setup(data=base_train, target="shot_made_flag", session_id=42, log_experiment=False)
        decision_tree_model = exp.create_model("dt")

        preds = exp.predict_model(decision_tree_model, data=X_test)
        y_pred_prob = preds["prediction_score"]
        y_pred_label = preds["prediction_label"]

        mlflow.log_param("Modelo", "Árvore de Decisão")
        mlflow.log_metric("Log Loss", log_loss(y_test, y_pred_prob))
        mlflow.log_metric("F1 Score", f1_score(y_test, y_pred_label))
        mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred_label))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pkl")
            joblib.dump(decision_tree_model, model_path)

            signature = infer_signature(X_test, decision_tree_model.predict_proba(X_test))
            input_example = X_test.head(3)

            mlflow.pyfunc.log_model(
                artifact_path="Modelo_Arvore",
                python_model=SklearnProbaWrapper(),
                artifacts={"model_path": model_path},
                signature=signature,
                input_example=input_example,
            )

        return decision_tree_model
