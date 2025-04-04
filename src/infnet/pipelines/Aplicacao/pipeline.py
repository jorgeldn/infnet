from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                nodes.aplicar_modelo_producao,
                inputs="dataset_kobe_prod",
                outputs="resultados_aplicacao",
                name="aplicar_modelo_producao"
            )
        ]
    )
