#!/bin/bash
#SBATCH --job-name=train_brats_residual_up_idwt_dec_multilevel_no_cb
#SBATCH --account=brain-lab
#SBATCH --output=/blue/brain-lab/mdmahfuzalhasan/scripts/waveformer-brats/results/train_brats_residual_up_idwt_dec_multilevel_no_cb.%J.out
#SBATCH --error=/blue/brain-lab/mdmahfuzalhasan/scripts/waveformer-brats/results/train_brats_residual_up_idwt_dec_multilevel_no_cb.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:4
#SBATCH --time=24:00:00

module load conda
conda activate miccai

cd /blue/brain-lab/mdmahfuzalhasan/scripts/waveformer-brats

# Execute the Python script
srun python 3_train.py