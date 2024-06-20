#!/bin/bash

#SBATCH --account=carney-ashenhav-condo
#SBATCH --time=700:00:00
#SBATCH --mem=48G
#SBATCH -n 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan_grahek@brown.edu
#SBATCH -J Model1_LFXC_703
#SBATCH -o R-%x.%j.out
module load anaconda/3-5.2.0
source activate pyHSSM 
python ../models/model1.py

