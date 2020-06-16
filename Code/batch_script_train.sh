#!/bin/bash

#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID
# --- Start of slurm commands -----------
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH --time=72:00:00
#SBATCH --mem=24G
# Specify a job name:
#SBATCH -J qm9-64-0.0001-64-d201-48walks-newembed
# Specify an output file
# %j is a special variable that is replaced by the JobID when
# job starts
#SBATCH -o qm9-64-0.0001-64-%j-d201-48walks-newembed.out
#SBATCH -e qm9-64-0.0001-64-%j-d201-48walks-newembed.out
#----- End of slurm commands ----
# User specific aliases and functions
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/jkijews1/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/jkijews1/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/users/jkijews1/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/jkijews1/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
module unload python
conda activate drugdesign2
python regression.py ../../scratch/qm9 64 0.0001 64 48 densenet201 qm9-64-d201-newembed