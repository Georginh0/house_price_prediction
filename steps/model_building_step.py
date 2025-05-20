import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
from mlflow.models.signature import infer_signature
from zenml.enums import ArtifactType

# Define the model properly as a ZenML Model object
prices_predictor = Model(
    name="prices_predictor",
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(
    enable_cache=False,
    experiment_tracker=Client().active_stack.experiment_tracker.name,
    model=prices_predictor,  # Pass the Model object, not string
)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[
    Pipeline, ArtifactConfig(name="sklearn_pipeline", artifact_type=ArtifactType.MODEL)
]:
    """Builds and trains a Linear Regression model with preprocessing."""

    # Input validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify column types
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Create preprocessing pipelines
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Create model pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )

    # Set MLflow tracking URI from ZenML
    mlflow.set_tracking_uri(Client().active_stack.experiment_tracker.get_tracking_uri())

    # Clean up any existing runs
    if mlflow.active_run():
        mlflow.end_run()

    # Start and manage MLflow run
    with mlflow.start_run() as run:
        # Enable autologging
        mlflow.sklearn.autolog()

        logging.info("Training Linear Regression model...")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Explicit model logging with all required parameters
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=prices_predictor.name,  # Use the Model object's name
            signature=signature,
            input_example=X_train.iloc[:1],
        )

        # Log expected columns if categorical features exist
        if len(categorical_cols) > 0:
            try:
                onehot_encoder = (
                    pipeline.named_steps["preprocessor"]
                    .transformers_[1][1]
                    .named_steps["onehot"]
                )
                expected_columns = numerical_cols.tolist() + list(
                    onehot_encoder.get_feature_names_out(categorical_cols)
                )
                mlflow.log_dict(
                    {"expected_columns": expected_columns}, "expected_columns.json"
                )
                logging.info(f"Model expects columns: {expected_columns}")
            except Exception as e:
                logging.warning(f"Could not log expected columns: {str(e)}")

    return pipeline
