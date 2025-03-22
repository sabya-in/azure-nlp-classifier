from azureml.core import Workspace

# Connect to existing Azure ML workspace
ws = Workspace.from_config()
print("Workspace connected:", ws.name)

