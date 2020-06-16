#!/bin/bash
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID
# --- Start of slurm commands -----------
#SBATCH --time=48:00:00
#SBATCH -n 32
#SBATCH --mem=128G
# Specify a job name:
#SBATCH -J test_submit
# Specify an output file
# %j is a special variable that is replaced by the JobID when
# job starts
#SBATCH -o test_submit-%j.out
#SBATCH -e test_submit-%j.out
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
python create_walks.py ../../../scratch/tox21 32 2 64