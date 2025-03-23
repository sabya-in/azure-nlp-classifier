from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, ComputeTarget
import re

# Connect to Azure ML
ws = Workspace.from_config()
exp = Experiment(ws, "mood-lstm-training")

# Define Compute Cluster (Serverless)
compute_name = "serverless-compute"

if compute_name not in ws.compute_targets:
    from azureml.core.compute import AmlCompute
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2",  # Choose an appropriate VM size
        max_nodes=4,  # Auto-scales up to 4 nodes
        idle_seconds_before_scaledown=240  # Auto-scale down after 4 min idle
    )
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)
else:
    compute_target = ws.compute_targets[compute_name]

# Lists all available environments
envs = Environment.list(ws)
tensorflow_env = [env for env in envs if re.search('TensorFlow', env, re.IGNORECASE)]

# Use a Pre-built Azure ML Environment
env = Environment.get(ws, name=f"{tensorflow_env}")  # Change based on type of imports used

# Configure Training Job
script_config = ScriptRunConfig(
    source_directory="./",
    script="train.py",
    arguments=[
        "--data_path", "data/mood_data.csv",
        "--output_dir", "outputs/"
    ],
    compute_target=compute_target,  # Serverless compute target
    environment=env
)

# Submit Job
run = exp.submit(script_config)
run.wait_for_completion(show_output=True)

