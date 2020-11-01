# instructions to run

## install nvidia-docker2
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## start up docker container
docker build -t cs269 .  
docker run -ti --gpus all --shm-size=1g --ulimit memlock=-1 -e   "TERM=xterm-256color" -v "$(pwd):/workspace" -p 8888:8888 -p 7722:7722 -p 6006:6006 cs269  
note that docker run automatically starts jupyter and tensorboard

## to ssh into container
ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" root@localhost -p 7722

## to connect to jupyter notebook, copy below into browser url
localhost:8888

## to connect to tensorboard
localhost:6006