from azureml.core import Workspace, Datastore

# Connect to Azure ML Workspace
ws = Workspace.from_config()

# Register ADLS as a datastore
datastore = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name="adls_datastore",
    container_name="azure-ml",
    account_name="datalakeintake",
    account_key="KCgqf/kGIOF9o/vw3u6ky3LcbKMdNlmQZq9n2wMKZrmQEFXa4GniENHqc+IPxOqPDhYQkFmdNTsb+AStTH8DjA=="
)

print("ADLS registered successfully:", datastore.name)

