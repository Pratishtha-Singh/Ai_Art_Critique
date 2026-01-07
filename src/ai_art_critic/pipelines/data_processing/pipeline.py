from kedro.pipeline import Pipeline, node
from .nodes import merge_datasets, generate_text_embeddings, generate_image_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=merge_datasets,
            inputs=["WikiArt", "Artemis"],
            outputs="unified_art_dataset",
            name="merge_datasets_node"
        ),
        node(
            func=generate_text_embeddings,
            inputs=["unified_art_dataset", "params:text_sample_size"],
            outputs="text_embedded_art_dataset",
            name="generate_text_embeddings_node"
        ),
        node(
            func=generate_image_embeddings,
            inputs=["text_embedded_art_dataset", "params:image_sample_size"],
            outputs="embedded_art_dataset",
            name="generate_image_embeddings_node"
        )
    ])