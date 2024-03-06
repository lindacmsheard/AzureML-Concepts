# Intro to AML Developer Platform

This repo walks through some examples of interacting with Azure Machine Learning with the Azure ML CLI V2 and its associated YAML schema syntax.


## 00 Pre-requisities
The required infrastructure is only an Azure resource group, and contributor permission to it. 

Provision this via any of the following routes 
- the [Azure portal](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal#create-resource-groups),
- the [azure cli](./infrastructure/cli)
- [bicep](./infrastructure/bicep) templating language
- or [terraform](./infrastructure/terraform) with deployment state management.


## 01 Getting started with the v2 CLI and YAML schemas

Start here: 
- [CLI-demo](./CLI-demo/v2.3.x/CLI-demo.md) for a basic walk through of features as of v2.3.x
- [CLI v2 in a day](https://github.com/caiomsouza/aml-cli-v2-in-a-day) for a more in-depth tutorials


<hr />

## Resources

- [CLI V2 docs](https://docs.microsoft.com/en-us/cli/azure/ml?view=azure-cli-latest)
- [Reference YAML schemas](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-overview)
- [github: azureml-examples - cli](https://github.com/Azure/azureml-examples/tree/main/cli)

- [VSCode Extension](https://techcommunity.microsoft.com/t5/azure-ai/supercharge-azure-ml-code-development-with-new-vs-code/ba-p/2260129)

- [AI School](https://www.microsoft.com/en-us/ai/ai-school)
- [AI for Beginners](https://dev.to/azure/announcing-a-new-free-curriculum-machine-learning-for-beginners-1h58)


