#! /usr/bin/zsh

source ports.sh

service ssh restart > /dev/null
#tmux new-session -d jupyter lab --no-browser --ip=0.0.0.0 --port=$JUPYTER_PORT --allow-root --NotebookApp.token="asdf" --notebook-dir=/workspace
tmux new-session -d tensorboard --port $TENSORBOARD_PORT --logdir /workspace/tensorboard
