from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment

# Connect to Azure ML
ws = Workspace.from_config()
exp = Experiment(ws, "mood-lstm-training")

# Define Environment
env = Environment.from_conda_specification(name="tensorflow-env", file_path="environment.yml")

# Define Training Job
script_config = ScriptRunConfig(
    source_directory="./",
    script="train.py",
    arguments=[
        "--data_path", "data/mood_data.csv",
        "--output_dir", "outputs/"
    ],
    compute_target="YOUR-AZURE-VM-NAME",
    environment=env
)

# Submit Job
run = exp.submit(script_config)
run.wait_for_completion(show_output=True)

