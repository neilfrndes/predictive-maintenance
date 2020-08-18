#!/bin/bash

# Activate environment and install packages
conda init bash
conda activate tensorflow2_latest_p37
source activate tensorflow2_latest_p37
pip install -r requirements.txt

# Download dataset
mkdir data
wget -nc http://azuremlsamples.azureml.net/templatedata/PM_train.txt -O ./data/train.csv
wget -nc http://azuremlsamples.azureml.net/templatedata/PM_test.txt -O ./data/test.csv
wget -nc http://azuremlsamples.azureml.net/templatedata/PM_truth.txt -O ./data/truth.csv

# Run inference
cd src
python run.py
