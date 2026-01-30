#!/bin/bash
#SBATCH --job-name=brats-waveformer
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mdmahfuzalhasan@ufl.edu
#SBATCH --account=brain-lab
#SBATCH --qos=brain-lab
#SBATCH --output=/blue/brain-lab/mdmahfuzalhasan/scripts/WaveFormer/results/brats_compressed_4x4.%J.out
#SBATCH --error=/blue/brain-lab/mdmahfuzalhasan/scripts/WaveFormer/results/brats_compressed_4x4.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:2
#SBATCH --time=30:00:00
pwd; hostname; date

module load conda
conda activate miccai

cd /blue/brain-lab/mdmahfuzalhasan/scripts/WaveFormer

# Execute the Python script
python 3_train.py
date