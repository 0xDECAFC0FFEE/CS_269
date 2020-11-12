# instructions to run

## install nvidia-docker2
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## start up slimmed docker container (smaller memory footprint, no jupyter, no gpu, 3.92GB)
docker build -t cs269_slim -f slim.dockerfile .  
docker run -ti --shm-size=1g -e "TERM=xterm-256color" -v "$(pwd):/workspace" -p 8888:8888 -p 7722:7722 -p 6006:6006 cs269_slim  

## start up full gpu docker container (13.8GB)
docker build -t cs269_full -f full.dockerfile .  
docker run -ti --gpus all --shm-size=1g --ulimit memlock=-1 -e "TERM=xterm-256color" -v "$(pwd):/workspace" -p 8888:8888 -p 7722:7722 -p 6006:6006 cs269_full  
note that docker run automatically starts jupyter and tensorboard  

## to ssh into container
ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" root@localhost -p 7722

## to connect to jupyter notebook, copy below into browser url
localhost:8888

## to connect to tensorboard
localhost:6006