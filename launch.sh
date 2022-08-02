#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=myJob.err
#SBATCH --output=myJob.out
#SBATCH --gres=gpu:1
#SBATCH --partition=g100_usr_interactive
#SBATCH --account=uBS21_InfGer_0
#SBATCH --time=00:20:00
#SBATCH --mem=32G



./training-latplan2.py

#./collective.py

#./deduction.py

#./manifold.py

#./missing.py

#./reasoning.py
