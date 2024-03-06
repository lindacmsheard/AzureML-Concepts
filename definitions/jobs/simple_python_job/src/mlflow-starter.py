# imports
import sys
import os
import mlflow

from random import random

# define functions
def main():
    # some other code related to ML
    mlflow.log_param("my_custom", "my value for Hyosun")
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("hello_param", "world")
    mlflow.log_metric("hello_metric", random())
    mlflow.log_metric("hello_metric", random())
    mlflow.log_metric("hello_metric", random())
    mlflow.log_metric("hello_metric", random())

    os.system(f"echo 'hello world' > helloworld.txt")
    mlflow.log_artifact("helloworld.txt")


# run functions
if __name__ == "__main__":
    # run main function
    main()