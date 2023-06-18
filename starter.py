#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys

from datetime import datetime

import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        (dv, model) = pickle.load(f_in)
        
    return dv, model

def read_data(filename, year, month):
    categorical = ['PULocationID', 'DOLocationID']

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']

    dicts = df[categorical].to_dict(orient='records')
    return dicts


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    
    df_result["ride_id"] = df["ride_id"]
    df_result["predictions"] = y_pred

    taxi_type = output_file.split("/")[1]
    path_directory = f'output/{taxi_type}'
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def apply_model(input_file, output_file, taxi_type, year, month):
    model_path = 'model.bin'
    
    print(f"------ Reading the data from {input_file} ------")
    df = read_data(input_file, year, month)
    dicts = prepare_dictionaries(df)

    print(f"------ Loading the model from {model_path} ------")
    dv, model = load_model(model_path)

    print("------ Applying the model ------")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print("The std  of the predictions is: %.2f" % y_pred.std())
    print("The mean of the predictions is: %.2f" % y_pred.mean())

    print(f"------ Saving the result in {output_file} ------")
    save_results(df, y_pred, output_file)
    
    return output_file

def get_paths(taxi_type, year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    return input_file, output_file

def ride_duration_prediction(taxi_type, year, month):
    input_file, output_file = get_paths(taxi_type, year, month)

    apply_model(
        input_file=input_file,
        output_file=output_file,
        taxi_type=taxi_type, 
        year=year,
        month=month
    )

    print(f"------ Get file size from {output_file} ------")
    file_stats = os.stat(output_file)
    print(f'File Size in Bytes is {file_stats.st_size:.2f}')
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024):.2f}')

def run():
    taxi_type = sys.argv[1]  #"yellow"
    year = int(sys.argv[2])  # 2022
    month = int(sys.argv[3])  #2

    ride_duration_prediction(
        taxi_type=taxi_type,
        year=year,
        month=month
    )    


if __name__ == '__main__':
    run()