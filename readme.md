# instructions to run

## install nvidia-docker2
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## start up docker container
docker build -t cs269 .
docker run -td --gpus all --shm-size=1g --ulimit memlock=-1 -e "TERM=xterm-256color" -v $(pwd):"/workspace" -p 8888:8888 -p 7722:7722 cs269

## ssh into container
ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" root@localhost -p 7722

## start jupyter notebook locally
tmux-wrap jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=""

## to connect to notebook, copy below into browser url
localhost:8888