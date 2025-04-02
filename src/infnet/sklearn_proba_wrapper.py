import mlflow.pyfunc
import joblib

class SklearnProbaWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)
