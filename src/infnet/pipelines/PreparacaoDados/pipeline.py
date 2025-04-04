"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.prepare_data,
            inputs=['dataset_kobe_dev'],
            outputs='data_filtered',
            tags=['preprocessamento']
        ),
        node(
            nodes.split_bases,
            inputs=['data_filtered'],
            outputs=['base_train', 'base_test'],
            tags=['separacao']
        )
    ])
