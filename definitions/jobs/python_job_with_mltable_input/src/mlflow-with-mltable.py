# imports
import sys
import os
import mlflow
import mltable

from random import random

import argparse

import pandas as pd


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--iris-csv", type=str)
    parser.add_argument("--my-mltable", type=str)

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



    # log some info about our mltable input - expect only the MLTable file to reside here
    
    files = os.listdir(args.my_mltable)

    with open('user_logs/mylog.txt', 'w') as f:
        f.write(f'local mount: {args.my_mltable}\n')
        f.write(f"{files}")

    # load data with mltable and log evidence of reading the data

    tbl = mltable.load(args.my_mltable)

    df = tbl.to_pandas_dataframe()

    with open('user_logs/mylog.txt', 'a') as f:
        f.write('\n\n\n======================')
        f.write('\nMLTable -> df.head(5):')
        f.write(f'(Dataset: {len(df)} rows)')
        f.write('\n======================\n')
        f.write(str(df.head(5)))



# run functions
# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)