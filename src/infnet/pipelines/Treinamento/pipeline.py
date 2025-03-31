"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.train_models,
            inputs=["base_train", "base_test"],
            outputs=None,
            tags=['treinamento']
        ),
    ])
