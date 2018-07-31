#!/bin/bash

#SBATCH -t 6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=40
#SBATCH -J gp_noisy-peak                                    # job name
#SBATCH --mem=32G                                     # memory in GB

# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=zq@princeton.edu

module load anaconda3 intel intel-mkl
export OMP_NUM_THREADS=40
python -u gp.py -o /tigress/zequnl/gp_chains/5z_Peaks_low_${SLURM_JOB_ID}.dat \
  -d Peaks -binmin -0.8 -binmax 1 -mn KN -cn KN --bin_center_row 1
