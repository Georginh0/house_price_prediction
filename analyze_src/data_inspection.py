from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract Base Class for Data Inspection Strategies
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> dict:
        """
        Perform a specific type of data inspection.

        Parameters:
            df (pd.DataFrame): The dataframe on which the inspection
                is to be performed.

        Returns:
            dict: A dictionary containing the inspection results.
        """
        pass


# Concrete Strategy for Data Types Inspection
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> dict:
        """
        Inspect the data types of each column and count non-null values.

        Parameters:
            df (pd.DataFrame): The dataframe on which the inspection
                is to be performed.

        Returns:
            dict: A dictionary containing data types and non-null counts.
        """
        logging.info("Inspecting data types and non-null counts...")
        df.info()
        return {"data_types": df.dtypes, "non_null_counts": df.notnull().sum()}


# Concrete Strategy for Summary Statistics Inspection
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> dict:
        """
        Provide summary statistics for numerical and categorical features.

        Parameters:
            df (pd.DataFrame): The dataframe on which the inspection
                is to be performed.

        Returns:
            dict: A dictionary containing summary statistics for numerical
            and categorical features.
        """
        logging.info("Generating summary statistics...")
        summary_all = df.describe(include="all")
        numerical_stats = summary_all.loc[:, summary_all.loc["count"].notnull()]
        categorical_stats = summary_all.loc[:, summary_all.loc["unique"].notnull()]
        return {
            "numerical_summary": numerical_stats,
            "categorical_summary": categorical_stats,
        }


# Context Class that uses a DataInspectionStrategy
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
            strategy (DataInspectionStrategy): The strategy to be used for
            data inspection.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Set a new inspection strategy.

        Parameters:
            strategy (DataInspectionStrategy): The new strategy to be used for
            data inspection.
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame) -> dict:
        """
        Executes the inspection using the current strategy.

        Parameters:
            df (pd.DataFrame): The dataframe to be inspected.

        Returns:
            dict: The inspection results returned by the strategy.
        """
        return self._strategy.inspect(df)


if __name__ == "__main__":
    # Load the data using pathlib for path handling
    data_path = Path(
        "/Users/georgensamuel/Documents/Machine_Learning_Projects/"
        "house_price_prediction/extracted_data/AmesHousing.csv"
    )
    df = pd.read_csv(data_path)

    # Initialize the Data Inspector with a specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    data_types_results = inspector.execute_inspection(df)
    logging.info("Data Types Inspection Results: %s", data_types_results)

    # Change strategy to Summary Statistics and execute
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    summary_stats_results = inspector.execute_inspection(df)
    logging.info("Summary Statistics Inspection Results: %s", summary_stats_results)
