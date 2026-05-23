#!/bin/bash

# export ENROOT_DATA_PATH="/raid/tmp/enroot_data_nk"
# export ENROOT_TEMP_PATH="/raid/tmp/enroot_temp_nk"
enroot create -n test /mnt/weka/aisg/sqsh/nemo:25.11.sqsh
enroot start --rw test bash -c "cd ${SLURM_SUBMIT_DIR} && source /opt/venv/bin/activate && exec bash"
