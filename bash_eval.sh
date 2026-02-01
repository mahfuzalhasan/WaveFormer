#!/bin/bash
#SBATCH --job-name=brats-waveformer_2x2-eval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mdmahfuzalhasan@ufl.edu
#SBATCH --account=brain-lab
#SBATCH --qos=brain-lab
#SBATCH --output=/blue/brain-lab/mdmahfuzalhasan/scripts/WaveFormer/results/brats_compressed_2x2_eval.%J.out
#SBATCH --error=/blue/brain-lab/mdmahfuzalhasan/scripts/WaveFormer/results/brats_compressed_2x2_eval.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
pwd; hostname; date

module load conda
conda activate miccai

cd /blue/brain-lab/mdmahfuzalhasan/scripts/WaveFormer

# Execute the Python script
python 5_compute_metrices.py
date