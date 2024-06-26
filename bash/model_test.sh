#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH -n 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yiming_zhao1@brown.edu
#SBATCH -J Model_testing
#SBATCH -o R-%x.%j.out
module load anaconda/3-5.2.0
source activate pyHSSM 
python ../models/model_test.py