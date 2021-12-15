source ${CONDA_PATH}

if conda info --envs | grep "${tf_venv}"; then
  conda activate "${tf_venv}"
else
  conda create --name "${tf_venv}" python=3.8
  conda activate "${tf_venv}"
  conda install tensorflow
fi