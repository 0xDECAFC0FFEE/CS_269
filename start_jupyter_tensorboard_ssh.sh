#! /usr/bin/zsh

service ssh restart
tmux new-session -d jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token="" --notebook-dir=/workspace
tmux new-session -d tensorboard --port 6006 --logdir /workspace/tensorboard --bind_all