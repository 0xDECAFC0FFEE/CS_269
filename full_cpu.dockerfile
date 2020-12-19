FROM pytorch/pytorch

# update system packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential wget tmux nmap vim htop unzip

# install zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)"
RUN chsh -s `which zsh`

# set up ssh
RUN apt-get -y install openssh-server
RUN echo "root ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN passwd -d `whoami`
COPY ports.sh ports.sh
RUN echo "Port "`./ports.sh ssh`"\nPermitEmptyPasswords yes\nX11Forwarding yes\nPrintMotd no\nAcceptEnv LANG LC_*\nSubsystem       sftp    /usr/lib/openssh/sftp-server\nPasswordAuthentication yes\nPermitRootLogin yes" > /etc/ssh/sshd_config

# installing notebook tqdm for jupyter
RUN conda install jupyterlab -y
RUN conda install -c conda-forge ipywidgets -y
RUN conda upgrade -c conda-forge jupyterlab -y
RUN conda install nodejs -y
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager -y

# install lucas's env
ADD https://api.github.com/repos/0xDECAFC0FFEE/.setup/git/refs/ version.json
RUN git clone https://github.com/0xDECAFC0FFEE/.setup.git /root/.setup --recursive
RUN python3 /root/.setup/setup.py --disable-ssh

# installing project requirements.txt
RUN conda config --append channels conda-forge
RUN conda install pytorch torchvision cpuonly -c pytorch
RUN conda install numpy ninja cffi typing_extensions future dataclasses tqdm
RUN jupyter lab build
RUN conda install expect gdown snakeviz

CMD ./start_jupyter_tensorboard_ssh.sh && cd /workspace && `which zsh`
