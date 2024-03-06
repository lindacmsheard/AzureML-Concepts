# Azure pre-req

Ensure you have a resource group to work with, and have contributor permissions. 

(see [00 - Pre-requisites](../Readme.md) here)



# Azure CLI pre-req

From a terminal on your local machine, or from the terminal of an Azure ML compute instance:

[Install the Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) or verify existing cli version
```
az version
```

> if the Azure CLI is before v2.11.0, go through a [full install](https://docs.microsoft.com/en-us/cli/azure/update-azure-cli) to upgrade. 

Otherwise, run 
```
az upgrade
```
Ensure the az version is greater than 2.15



### Further reading: 
- https://docs.microsoft.com/en-gb/azure/machine-learning/how-to-configure-cli 

# AZ ML cli extension
make sure to remove [v1 - still  in the docs here](https://docs.microsoft.com/en-us/cli/azure/ml(v1)?view=azure-cli-latest)
```
az extension remove -n azure-cli-ml
```

Install the ml v2 extension ([docs](https://docs.microsoft.com/en-us/cli/azure/ml?view=azure-cli-latest)):
```
az extension add -n ml
```

or update it:
```
az extension update -n ml
```

Review the available subcommands with 
```
az ml -h
```

### Further Reading:
- https://docs.microsoft.com/en-gb/azure/machine-learning/how-to-configure-cli 


# CLI defaults

Once available, set the default resource group and location, to avoid having to pass these with every CLI command:

The example here sets the defaults locally, applying within the current folder and subfolders only. (Note that a workspace with the given name does not have to exist yet)
```
az configure --scope local --defaults group=<resource group> location=uksouth workspace=amlv2test
az configure --scope local --list-defaults
```

or in the newer version:
```
az config set defaults.location=uksouth
az config set --local defaults.group=<resource group> defaults.workspace=<workspace>
```

> Tip: if working on an Azure ML Compute Instance, change into your working directory at `/home/azureuser/cloudfiles/code/Users/<yourusername>`. Everything inside of `code` is mounted from Azure Storage, and therefore can persist beyond the lifetime of this particular compute instance. This includes defaults set with the --local option from within that directory.

# Workspace
*https://azuremlschemas.azureedge.net/latest/workspace.schema.json*

(Ignore this section if already working from an Azure ML compute instance in an Azure ML workspace)

> 💡 remember that the resource group parameter is omitted - it's set by the defaults configured in the pre-req section

> ⚠️ the workspace name set in the yaml file will be overridden by the cli parameter `-w`, or by what is set in the cli defaults. Remember to check `az configure --list-defaults`

```sh
# az ml workspace create
az ml workspace create --file workspace.yml
```

verify workspace creation with:
```
az resource list -o table
```

### Further Reading
- https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-overview 


# Compute
```
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 10 
az ml compute create -f compute/compute-gpu.yml
```


```
az ml compute create -f compute/compute-ci.yml
```
> ⚠️Review the Yaml schema documentation at https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-overview for changes to yaml schemas

List available compute:

instances:
```
az ml compute list --type computeinstance -o table
```
clusters:
```
az ml compute list --type AmlCompute -o table
```

Start/stop compute instance
```
az ml compute start -n myDS2
az ml compute stop -n myDS2
```



Connect to the a compute instance with the VSCode Azure Machine learning extension:

> ⚠️ If starting from wsl, ensure that the entension is also installed on the Windows side.

- Bring up the command pallette (CTRL + SHIFT + P) > `Azure ML: Connect to Compute Instance`
- Alternatively, just click the 'VSCode' button on the compute list entry in the Azure Machine learning studio, and follow the instructions. 

### Notes
- allowed values for type are not camel case (e.g. AmlCompute as type fails, use `amlcompute` and `computeinstance`)

# Data

```
az ml data create -f datasets/dataset2.yml
```

# Environments

```
az ml environment create -f envs/<env-def>.yml
```
#### Ref
- https://docs.microsoft.com/en-us/cli/azure/ml/environment?view=azure-cli-latest
- https://github.com/Azure/azureml-examples/tree/main/cli/assets/environment
- https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-environment
- https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-overview 
- https://mcr.microsoft.com/v2/_catalog

# Jobs

Command jobs
```
az ml job create -f jobs/job.yml
```

integrating with data and mlflow
```
az ml job create -f jobs/r-job.yml
```




## Further reading:
- https://docs.microsoft.com/en-gb/azure/machine-learning/how-to-train-cli

