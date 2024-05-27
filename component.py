from kubernetes.client import V1EnvVar, V1Volume, V1VolumeMount, V1PersistentVolumeClaimVolumeSource
import pandas as pd
from kfp.dsl import component, InputPath, OutputPath
from kfp import dsl


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "mlflow"]
)
def preprocess(file_path: InputPath("CSV"), output_file: OutputPath("CSV")):
    import pandas as pd
    
    header = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    df = pd.read_csv(file_path, header=None, names=header)
    # encode categorical data as integers
    categorical_columns = [
        "age",
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
        "income",
    ]
    df[categorical_columns] = df[categorical_columns].apply(
        lambda x: x.astype("category").cat.codes, axis=0
    )
  
    df.to_csv(output_file, index=False)


@component(
    base_image="python:3.9",
    packages_to_install=["mlflow", "pandas", "scikit-learn", "boto3"]
)
def train(file_path: InputPath("CSV")) -> str:
    import mlflow
    import mlflow.data
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import boto3
    from mlflow.models import infer_signature


    mlflow.set_tracking_uri("http://mlflow-server.local/")

    df = pd.read_csv(file_path)

    labels_column = "income"
    train_x, test_x, train_y, test_y = train_test_split(
        df.drop([labels_column], axis=1), 
        df[labels_column], 
        random_state=69
    )

    dataset = mlflow.data.from_pandas(df, source=file_path)

        

    with mlflow.start_run(run_name="income_training"):
        alpha, hidden_layers = 1e-3, (2, 3)
        mlp = MLPClassifier(
            solver="lbfgs",
            alpha=alpha,
            hidden_layer_sizes=hidden_layers,
            random_state=69,
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("hidden_layers", hidden_layers)

        mlp.fit(train_x, train_y)

        preds = mlp.predict(test_x)

        signature = infer_signature(train_x, preds)

        accuracy = (test_y == preds).sum() / preds.shape[0]
        mlflow.log_metric("accuracy", accuracy)

        reg_model_name = "sk_income_model"
        # Log the trained model artifact
        result = mlflow.sklearn.log_model(
            sk_model=mlp,
            artifact_path="income_model",  # Corrected relative path within the bucket
            signature=signature,
            registered_model_name=reg_model_name,
        )

        # Log additional artifacts
        # For example, log a confusion matrix plot
        fig, ax = plt.subplots()
        # Plot confusion matrix here
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Save any additional files you want to log as artifacts
        with open("additional_file.txt", "w") as f:
            f.write("Additional artifact content")
        mlflow.log_artifact("additional_file.txt")

        return f"{mlflow.get_artifact_uri()}/{result.artifact_path}"


@component(
    base_image="python:3.9",
    packages_to_install=["requests", "mlflow"]
    
)
# Define the function for downloading the dataset
def download_dataset(url: str, output_file: OutputPath()):
    import requests  # Import requests inside the function

    # Use requests.get() to download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with open(output_file, 'wb') as f:
        f.write(response.content)


# Create the component from the function and specify volume and volume mount



@dsl.pipeline(
    name="income_pipeline",
    description="Pipeline for training and deploying a model trained on Census Income dataset",
)
def income_pipeline():
    downloader_task = download_dataset(url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
    downloader_task.set_caching_options(False)
    # Mount the PVC to the preprocess component
    preprocess_task = preprocess(file_path = downloader_task.output)
    preprocess_task.set_caching_options(False)

    train_task = (
    train(file_path=preprocess_task.output)
    .set_env_variable(
        name="MLFLOW_TRACKING_URI",
        value="http://mlflow-server.local/",  # Adjust service name
    )
    .set_env_variable(
        name="MLFLOW_S3_ENDPOINT_URL",
        value="http://mlflow-minio.local:30869/",  # Adjust service name and port
    )
    .set_env_variable(
        name="AWS_ACCESS_KEY_ID",
        value="minio",
    )
    .set_env_variable(
        name="AWS_SECRET_ACCESS_KEY",
        value="minio123",
    )
    )
    train_task.set_caching_options(False)
    #train_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    

# Compile the pipeline
from kfp import compiler

compiler.Compiler().compile(pipeline_func=income_pipeline, package_path="income.yaml")
