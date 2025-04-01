"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.train_logistic_model,
            inputs=["base_train", "base_test"],
            outputs="lr_model",
            name="treinar_regressao_logistica",
        ),
        node(
            nodes.train_decision_tree_model,
            inputs=["base_train", "base_test"],
            outputs="dt_model",
            name="treinar_arvore_decisao",
        ),
    ])
