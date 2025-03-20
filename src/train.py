import azureml.core
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

# Load Azure ML Workspace
ws = Workspace.from_config()

# Define Compute Target
compute_target = ComputeTarget(workspace=ws, name="NLPCompute")

# Create Experiment
exp = Experiment(workspace=ws, name="NLP-Training")

# Define Training Script
script_config = ScriptRunConfig(
    source_directory=".",
    script="train.py",
    compute_target=compute_target
)

# Submit Experiment
run = exp.submit(script_config)
run.wait_for_completion(show_output=True)

