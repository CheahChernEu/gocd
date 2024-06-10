from kubernetes.client import V1EnvVar, V1Volume, V1VolumeMount, V1PersistentVolumeClaimVolumeSource
from kfp.dsl import component, InputPath, OutputPath
from kfp import dsl
import re
import requests
from urllib.parse import urlsplit

def get_istio_auth_session(url: str, username: str, password: str) -> dict:
    """
    Determine if the specified URL is secured by Dex and try to obtain a session cookie.
    WARNING: only Dex `staticPasswords` and `LDAP` authentication are currently supported
             (we default default to using `staticPasswords` if both are enabled)

    :param url: Kubeflow server URL, including protocol
    :param username: Dex `staticPasswords` or `LDAP` username
    :param password: Dex `staticPasswords` or `LDAP` password
    :return: auth session information
    """
    # define the default return object
    auth_session = {
        "endpoint_url": url,    # KF endpoint URL
        "redirect_url": None,   # KF redirect URL, if applicable
        "dex_login_url": None,  # Dex login URL (for POST of credentials)
        "is_secured": None,     # True if KF endpoint is secured
        "session_cookie": None  # Resulting session cookies in the form "key1=value1; key2=value2"
    }

    # use a persistent session (for cookies)
    with requests.Session() as s:

        ################
        # Determine if Endpoint is Secured
        ################
        resp = s.get(url, allow_redirects=True)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {url}"
            )

        auth_session["redirect_url"] = resp.url

        # if we were NOT redirected, then the endpoint is UNSECURED
        if len(resp.history) == 0:
            auth_session["is_secured"] = False
            return auth_session
        else:
            auth_session["is_secured"] = True

        ################
        # Get Dex Login URL
        ################
        redirect_url_obj = urlsplit(auth_session["redirect_url"])

        # if we are at `/auth?=xxxx` path, we need to select an auth type
        if re.search(r"/auth$", redirect_url_obj.path): 
            
            #######
            # TIP: choose the default auth type by including ONE of the following
            #######
            
            # OPTION 1: set "staticPasswords" as default auth type
            redirect_url_obj = redirect_url_obj._replace(
                path=re.sub(r"/auth$", "/auth/local", redirect_url_obj.path)
            )
            # OPTION 2: set "ldap" as default auth type 
            # redirect_url_obj = redirect_url_obj._replace(
            #     path=re.sub(r"/auth$", "/auth/ldap", redirect_url_obj.path)
            # )
            
        # if we are at `/auth/xxxx/login` path, then no further action is needed (we can use it for login POST)
        if re.search(r"/auth/.*/login$", redirect_url_obj.path):
            auth_session["dex_login_url"] = redirect_url_obj.geturl()

        # else, we need to be redirected to the actual login page
        else:
            # this GET should redirect us to the `/auth/xxxx/login` path
            resp = s.get(redirect_url_obj.geturl(), allow_redirects=True)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {redirect_url_obj.geturl()}"
                )

            # set the login url
            auth_session["dex_login_url"] = resp.url

        ################
        # Attempt Dex Login
        ################
        resp = s.post(
            auth_session["dex_login_url"],
            data={"login": username, "password": password},
            allow_redirects=True
        )
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials were probably invalid - "
                f"No redirect after POST to: {auth_session['dex_login_url']}"
            )

        # store the session cookies in a "key1=value1; key2=value2" string
        auth_session["session_cookie"] = "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    return auth_session

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

from kfp.client import Client


KUBEFLOW_ENDPOINT = "http://194.233.80.15:8080"
KUBEFLOW_USERNAME = "user@example.com"
KUBEFLOW_PASSWORD = "12341234"

auth_session = get_istio_auth_session(
    url=KUBEFLOW_ENDPOINT,
    username=KUBEFLOW_USERNAME,
    password=KUBEFLOW_PASSWORD
)


client = Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline", cookies=auth_session["session_cookie"])
# client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline", namespace="kubeflow-user-example-com", cookies=auth_session["session_cookie"])


client.create_run_from_pipeline_package('income.yaml', arguments={}, namespace="kubeflow-user-example-com")

