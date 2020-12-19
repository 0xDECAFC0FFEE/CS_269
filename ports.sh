#! /usr/bin/zsh

SSH_PORT=7622
JUPYTER_PORT=8899
TENSORBOARD_PORT=6007

if [ $# -eq 1 ]; then
	if [ $1 = "ssh" ]; then
		echo $SSH_PORT
	elif [ $1 = "jupyter" ]; then
		echo $JUPYTER_PORT
	elif [ $1 = "tb" ]; then
		echo $TENSORBOARD_PORT
	elif [ $1 = "docker" ]; then
		echo "-p $SSH_PORT:$SSH_PORT -p $JUPYTER_PORT:$JUPYTER_PORT -p $TENSORBOARD_PORT:$TENSORBOARD_PORT"
	fi
fi
