#!/bin/bash

AUG_FLAG=""
IP_FLAG=""

if [ "$1" -eq 1 ]; then
  AUG_FLAG="--use_augmentation"
fi

if [ "$2" -eq 1 ]; then
  IP_FLAG="--use_impact_point"
fi

python3 run.py $AUG_FLAG $IP_FLAG