import os
import pandas as pd
import zipfile
from abc import ABC, abstractmethod


# Define an abstract class for data ingestion
class DataIngestion(ABC):
    @abstractmethod
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP file ingestion
class ZipDataIngestion(DataIngestion):
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a ZIP file and return a pandas DataFrame."""

        # Check if the file is a ZIP archive
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Create the extraction folder if it doesn't exist
        os.makedirs("extracted_data", exist_ok=True)

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # List the extracted files
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        # Check if any CSV files were found
        if not csv_files:
            raise ValueError("No CSV files found in the ZIP archive.")

        # Load the first CSV file in the ZIP archive
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        data = pd.read_csv(csv_file_path)

        return data


# Implement a Factory to create DataIngestor
class DataIngestorFactory:
    @staticmethod
    def create_ingestor(file_extension: str) -> DataIngestion:
        """Factory method to create a DataIngestor based on the file type."""
        if file_extension == ".zip":
            return ZipDataIngestion()
        else:
            raise ValueError("Unsupported file type for data ingestion.")


if __name__ == "__main__":
    # Specify the file path
    file_path = "data/archive.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.create_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest_data(file_path)

    # Now df contains the DataFrame from the extracted CSV
    print(df.head())
