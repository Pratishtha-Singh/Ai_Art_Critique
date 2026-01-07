"""Custom Kedro dataset for loading data from Hugging Face Hub."""

import pandas as pd
from huggingface_hub import HfApi
from kedro.io.core import AbstractDataset, DatasetError


class HuggingFaceDataset(AbstractDataset):
    """Dataset for loading data from Hugging Face Hub.

    This dataset loads data from Hugging Face datasets and returns it as a pandas DataFrame.
    For datasets stored as parquet files, it downloads and concatenates them.
    """

    def __init__(
        self,
        dataset: str,
        split: str = "train",
        revision: str = "main",
        max_files: int = None,
        sample_size: int = None,
        **kwargs
    ):
        """Initialize the HuggingFaceDataset.

        Args:
            dataset: The Hugging Face dataset name (e.g., 'huggan/wikiart')
            split: The dataset split to load ('train', 'test', 'validation', etc.)
            revision: The specific revision/commit to load
            max_files: Maximum number of parquet files to load (for testing)
            sample_size: Maximum number of rows to load (for memory management)
            **kwargs: Additional arguments (unused for compatibility)
        """
        self.dataset = dataset
        self.split = split
        self.revision = revision
        self.max_files = max_files
        self.sample_size = sample_size
        self.api = HfApi()

    def _load(self) -> pd.DataFrame:
        """Load data from Hugging Face Hub."""
        try:
            # Get dataset info to verify it exists
            info = self.api.dataset_info(self.dataset)
            print(f"Loading dataset: {self.dataset} ({info.card_data.get('size_categories', ['unknown'])[0]})")

            # List files in the dataset
            files = self.api.list_repo_files(self.dataset, repo_type="dataset")

            # Filter for parquet files in the requested split
            parquet_files = [f for f in files if f.startswith(f"data/{self.split}") and f.endswith(".parquet")]
            parquet_files.sort()  # Ensure consistent ordering

            if not parquet_files:
                raise DatasetError(f"No parquet files found for split '{self.split}' in dataset '{self.dataset}'")

            # Limit files if specified
            if self.max_files:
                parquet_files = parquet_files[:self.max_files]
                print(f"Loading first {self.max_files} files out of {len(parquet_files)} total")

            print(f"Loading {len(parquet_files)} parquet files...")

            # Download and read each parquet file
            dfs = []
            for i, file_path in enumerate(parquet_files):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Loading file {i+1}/{len(parquet_files)}: {file_path}")

                # Download the file
                local_path = self.api.hf_hub_download(
                    self.dataset,
                    file_path,
                    repo_type="dataset",
                    revision=self.revision
                )

                # Read the parquet file
                df = pd.read_parquet(local_path)
                dfs.append(df)
       

            # Concatenate all dataframes
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Apply sample size limit if specified
                if self.sample_size and len(combined_df) > self.sample_size:
                    combined_df = combined_df.head(self.sample_size).copy()
                    print(f"Sampled dataset to {self.sample_size} rows for memory efficiency")
                
                print(f"Successfully loaded dataset with {len(combined_df)} rows and {len(combined_df.columns)} columns")
                return combined_df
            else:
                raise DatasetError("No data was loaded from the parquet files")

        except Exception as e:
            raise DatasetError(
                f"Failed to load dataset '{self.dataset}' from Hugging Face Hub. "
                f"Error: {str(e)}"
            ) from e

    def _save(self, data: pd.DataFrame) -> None:
        """Save data to Hugging Face Hub (not implemented)."""
        raise NotImplementedError(
            "Saving to Hugging Face Hub is not supported by this dataset."
        )

    def _exists(self) -> bool:
        """Check if the dataset exists on Hugging Face Hub."""
        try:
            self.api.dataset_info(self.dataset)
            return True
        except Exception:
            return False

    def _describe(self) -> dict:
        """Describe the dataset."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "revision": self.revision,
            "max_files": self.max_files,
            "sample_size": self.sample_size,
        }