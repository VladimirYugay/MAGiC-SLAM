#!/bin/bash
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --partition=capacity
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --time=6:00:00
#SBATCH --array=0-3

dataset="ReplicaMultiagent"
scenes=("office_0" "apart_0" "apart_1" "apart_2")
scenes_input_paths=("Office-0" "Apart-0" "Apart-1" "Apart-2")
INPUT_PATH="<path_to>/ReplicaMultiagent/"

OUTPUT_PATH="output"
CONFIG_PATH="configs/${dataset}"
EXPERIMENT_NAME="baseline"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}
SCENE_PATH=${scenes_input_paths[$SLURM_ARRAY_TASK_ID]}

source <path_to>/anaconda3/bin/activate
conda activate magic-slam
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job for dataset: $dataset, scene: $SCENE_NAME, experiment: $EXPERIMENT_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"


python run_slam.py "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --input_path "${INPUT_PATH}/${SCENE_PATH}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}"

echo "Job for scene $SCENE_NAME completed."
