#!/bin/bash -x
#SBATCH --job-name=waket
#SBATCH --partition=mpa
#SBATCH --nodes=1
#SBATCH --constraint="A100&GPUx1"
#SBATCH --time=5:00:00
#SBATCH --output=stdout
#SBATCH --error=stderr
#SBATCH --mail-type=END
#SBATCH --mail-user=alberto.martinez.de.la.ossa@desy.de

source $HOME/sim24.sh
python waket_script.py
