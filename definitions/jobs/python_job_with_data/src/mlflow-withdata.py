# imports
import sys
import os
import mlflow

from random import random

import argparse

import pandas as pd


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--iris-csv", type=str)
    parser.add_argument("--my-data", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args



# define functions
def main(args):
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("hello_param", "world")
    mlflow.log_metric("hello_metric", random())
    os.system(f"echo 'hello world' > helloworld.txt")
    mlflow.log_artifact("helloworld.txt")


    # read in data
    df = pd.read_csv(args.iris_csv)

    # print first 5 lines
    print(df.head(6))

    # ensure outputs directory exists 
    # The ./outputs and ./logs directories receive special treatment by Azure Machine Learning. 
    # If you write any files to these directories during your job, these files will get uploaded 
    # to the job so that you can still access them once the job is complete. The ./outputs 
    # folder is uploaded at the end of the job, while the files written to ./logs are uploaded 
    # in real time. Use the latter if you want to stream logs during the job, such as 
    # TensorBoard logs.

    #os.makedirs("outputs", exist_ok=True)
    #os.makedirs("mylogs", exist_ok=True)


    os.system(f"ls . > listdir.txt")
    mlflow.log_artifact("listdir.txt")


    # save data to outputs
    df.to_csv("outputs/iris.csv", index=False)



    # store some info about our file input
    
    files = os.listdir(args.my_data)

    with open('user_logs/mylog.txt', 'w') as f:
        f.write(f'local mount: {args.my_data}\n')
        f.write(f"{files}")





# run functions
# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)