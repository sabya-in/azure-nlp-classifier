from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandJob, PipelineJob
from azure.identity import DefaultAzureCredential
import os

# Authenticate with Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group=os.getenv("RESOURCE_GROUP"),
    workspace_name=os.getenv("WORKSPACE_NAME"),
)

# Define the preprocessing job
preprocess_job = CommandJob(
    code="./",
    command="python preprocess.py --data_path ${{inputs.raw_data}} --output_dir ${{outputs.processed_data}}",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",
    compute="cpu-cluster",
    inputs={"raw_data": "azureml:raw_training_data@latest"},
    outputs={"processed_data": "azureml://datastore/workspaceblobstore/preprocessed_data/"},
    experiment_name="ml_pipeline",
)

# Define the training job
train_job = CommandJob(
    code="./",
    command="python train.py --data_path ${{inputs.processed_data}} --output_dir ${{outputs.model_output}}",
    environment="AzureML-pytorch-1.10-ubuntu18.04-py37-cuda11.3:1",
    compute="gpu-cluster",
    inputs={"processed_data": preprocess_job.outputs.processed_data},
    outputs={"model_output": "azureml://datastore/workspaceblobstore/trained_model/"},
    experiment_name="ml_pipeline",
)

# Create the pipeline
@pipeline(name="ml_pipeline", description="Preprocess & Train Model")
def pipeline_func():
    preprocess_step = preprocess_job
    train_step = train_job

pipeline_job = pipeline_func()

# Submit the pipeline
ml_client.jobs.create_or_update(pipeline_job)
print("Pipeline submitted!")

