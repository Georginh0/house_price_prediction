import os
import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """python run_deployment.pypython run_deployment.py
    Ingest data from a ZIP file using the appropriate DataIngestor.

    Args:
        file_path (str): Path to the input ZIP file

    Returns:
        pd.DataFrame: The ingested data.
    """
    # Extract the file extension dynamically
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    return df
