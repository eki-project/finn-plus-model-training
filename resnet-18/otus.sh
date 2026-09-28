#!/bin/bash

__conda_setup="$('/opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh" ]; then
        . "/opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh"
    else
        export PATH="/opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate brevitasOtus

# Write the command line to be executed to the log
echo "$@"
# Forward all command line arguments as the command line to be run as the job
eval "$@"
