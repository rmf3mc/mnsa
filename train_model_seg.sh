#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide an argument (1-8)."
    exit 1
fi


# Get the argument
ARG=$1
model=$2
local_train=$3

echo "Model Number: $ARG"
echo "Backbone: $model"



if [ "$local_train" = true ]; then

    DATA_DIR="--data_dir ./data"
    Base_model="--model_path ./data/best_model.pth"
    Training_MC="--local_train 1"
else
    pip install wandb
    pip install optuna
    pip install plotly

    DATA_DIR="--data_dir ./data"
    Base_model="--model_path  ./data/best_model.pth"
    Training_MC="--local_train 0"
fi

echo "Where training: $local_train"


# Define the base command
BASE_CMD="python -u Train.py --mmanet --max_epoch 150"

# Run the command based on the argument
case $ARG in

    1)
        $BASE_CMD              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --unet $Training_MC  --transfer_to 0
        ;;

 
    2)
        $BASE_CMD    --fsds    --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model    --unet $Training_MC  --transfer_to 0

        ;;

    3)
        $BASE_CMD              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --unet $Training_MC  --transfer_to 0.125 
        ;;

 
    4)
        $BASE_CMD    --fsds    --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model    --unet $Training_MC  --transfer_to 0.125

        ;;

    5)
        $BASE_CMD              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --unet $Training_MC  --transfer_to 0.250 
        ;;

 
    6)
        $BASE_CMD    --fsds    --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model    --unet $Training_MC  --transfer_to 0.250 
        ;;


    7)
        $BASE_CMD              --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model   --unet $Training_MC  --transfer_to 0.50 
        ;;

 
    8)
        $BASE_CMD    --fsds    --seg_ild --freeze_all --dataparallel $DATA_DIR      --backbone_class  $model  $Base_model    --unet $Training_MC  --transfer_to 0.50 
        ;;

 

    *)
        echo "Invalid argument. Please provide a number between 1 and 10."
        exit 1
        ;;
esac

echo "Command executed."
