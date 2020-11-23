# instructions to run

## install nvidia-docker2
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## start up slimmed docker container (smaller memory footprint, no jupyter, no gpu, 3.92GB)
docker build -t cs269_slim -f slim.dockerfile .  
docker run -ti --shm-size=1g -e "TERM=xterm-256color" -v "$(pwd):/workspace" -p 8888:8888 -p 7722:7722 -p 6006:6006 -v /etc/localtime:/etc/localtime:ro cs269_slim  

## start up full gpu docker container (13.8GB)
docker build -t cs269_full -f full.dockerfile .  
docker run -ti --gpus all --shm-size=1g --ulimit memlock=-1 -e "TERM=xterm-256color" -v "$(pwd):/workspace" -p 8888:8888 -p 7722:7722 -p 6006:6006 -v /etc/localtime:/etc/localtime:ro cs269_full  
note that docker run automatically starts jupyter and tensorboard  

## to ssh into container
ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" root@localhost -p 7722

## to connect to jupyter notebook, copy below into browser url
localhost:8888

## to connect to tensorboard
localhost:6006

## Running on hoffman2
Hoffman2 does not support docker so you will need to manually set up the conda environment

### Setting up the conda environment
conda create -n env_name

conda config --append channels conda-forge

conda install jupyterlab -y

conda install -c conda-forge ipywidgets -y

conda upgrade -c conda-forge jupyterlab -y

conda install nodejs -y

conda install --file requirements_full.txt

conda install gdown -y

pip install rigl_torch

### Download jupyter notebook login script
Download login script

wget https://gitlab.idre.ucla.edu/dauria/jupyter-notebook/raw/master/h2jupynb

In h2jupynb add the following on line 367

pqsub.stdin.write('source activate env_name\n')

### Logging in to conda environment from jupyter notebook
python h2jupynb -u username -v anaconda3 -d Edge_Meta_Learning -t session_length_in_hours -m session_memory_in_GB -g yes -c V100
