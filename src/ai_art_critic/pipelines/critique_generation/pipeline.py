"""
This module defines the critique generation pipeline.
"""

from kedro.pipeline import Pipeline, node
from .nodes import create_art_critic_generator, generate_art_critique


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the critique generation pipeline.

    Returns:
        Kedro Pipeline object
    """
    return Pipeline([
        node(
            func=create_art_critic_generator,
            inputs=["params:index_path", "params:metadata_path"],
            outputs="art_critic_generator",
            name="create_generator"
        ),
        node(
            func=generate_art_critique,
            inputs=["art_critic_generator", "params:query"],
            outputs="critique_result",
            name="generate_critique"
        )
    ])