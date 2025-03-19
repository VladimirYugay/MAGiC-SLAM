#!/bin/bash
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=capacity
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --time=6:00:00
#SBATCH --array=0-1

dataset="AriaMultiagent"
scenes=("room0" "room1")
INPUT_PATH="<path_to>/Aria_Multiagent"

OUTPUT_PATH="output"
CONFIG_PATH="configs/${dataset}"
EXPERIMENT_NAME="baseline"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}

source <path_to>/anaconda3/bin/activate
conda activate magic-slam
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job for dataset: $dataset, scene: $SCENE_NAME, experiment: $EXPERIMENT_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"


python run_slam.py "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --input_path "${INPUT_PATH}/${SCENE_NAME}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}" \

echo "Job for scene $SCENE_NAME completed."
