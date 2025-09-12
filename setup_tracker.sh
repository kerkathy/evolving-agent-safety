#!/bin/bash

if [ "$1" == "baseline" ]; then
    mlflow server --backend-store-uri sqlite:///mydb1.sqlite --port 5001
elif [ "$1" == "main_method" ]; then
    mlflow server --backend-store-uri sqlite:///mydb.sqlite --port 5000 # original
else
    echo "Usage: $0 <baseline|main_method>"
    exit 1
fi
