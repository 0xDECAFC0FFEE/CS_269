# instructions to run

## Table of Contents
1. [Instructions to run Locally on Docker with GPU](#Instructions-to-run-Locally-on-Docker-with-GPU)
2. [Instructions to run on Raspberry Pi (for finetuning)](#Instructions-to-run-on-Raspberry-Pi-(for-finetuning))
3. [Instructions to run on Hoffman2](#Instructions-to-run-on-Hoffman2)
4. [Instructions to profile code](#Instructions-to-profile-code)

## Instructions to run Locally on Docker with GPU
### install nvidia-docker2
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### start up full gpu docker container (13.8GB)
docker build -t cs269_full -f full.dockerfile .  
docker run -ti --gpus all --shm-size=1g --ulimit memlock=-1 -e "TERM=xterm-256color" -v "$(pwd):/workspace" -p 8888:8888 -p 7722:7722 -p 6006:6006 -v /etc/localtime:/etc/localtime:ro cs269_full  
note that docker run automatically starts jupyter and tensorboard  

### to ssh into container
ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" root@localhost -p 7722

### to connect to jupyter notebook, copy below into browser url
localhost:8888

### to connect to tensorboard
localhost:6006

## Instructions to run on Raspberry Pi (for finetuning)

### install docker (standard, not nvidia-docker)

### start up slimmed docker container (smaller memory footprint, no jupyter, no gpu, 3.92GB)
docker build -t cs269_slim -f slim.dockerfile .  
docker run -ti --shm-size=1g -e "TERM=xterm-256color" -v "$(pwd):/workspace" -p 7722:7722 -v /etc/localtime:/etc/localtime:ro cs269_slim  

### to ssh into container
ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" root@localhost -p 7722

## Instructions to run on Hoffman2
Hoffman2 does not support docker so you will need to manually set up the conda environment

### Setting up the conda environment
qrsh -l h_rt=1:00:00,h_data=4G,gpu,RTX2080Ti

module load python/anaconda3
module load cuda/10.2
module load glibc/2.14

conda create --name EML
conda activate EML

conda install pytorch torchvision -c pytorch -y
conda config --append channels conda-forge
<!-- conda install numpy ninja cffi typing_extensions future dataclasses tqdm -y -->
conda install expect gdown snakeviz -y
pip install rigl_torch

### Download jupyter notebook login script
Download login script

wget https://gitlab.idre.ucla.edu/dauria/jupyter-notebook/raw/master/h2jupynb

### Logging in to conda environment from jupyter notebook
python h2jupynb -u username -v anaconda3 -d Edge_Meta_Learning -t session_length_in_hours -m session_memory_in_GB -g yes -c V100

## Instructions to profile code
docker run ... 
python3 -m cProfile -o results.prof main.py  
snakeviz results.prof -p 4433 -s -H 0.0.0.0  
go to localhost:4433/snakeviz/%2Fworkspace%2Fresults.prof in browser  