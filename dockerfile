FROM pytorch/pytorch

# update system packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential wget tmux nmap vim

# install zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)"
RUN chsh -s `which zsh`

# set up ssh
RUN apt-get -y install openssh-server
RUN echo "root ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN passwd -d `whoami`
RUN echo "Port 7722\nPermitEmptyPasswords yes\nX11Forwarding yes\nPrintMotd no\nAcceptEnv LANG LC_*\nSubsystem       sftp    /usr/lib/openssh/sftp-server\nPasswordAuthentication yes\nPermitRootLogin yes" > /etc/ssh/sshd_config
EXPOSE 7722

# install lucas's env
ADD https://api.github.com/repos/0xDECAFC0FFEE/.setup/git/refs/ version.json
RUN git clone https://github.com/0xDECAFC0FFEE/.setup.git /root/.setup
RUN python3 -m pip install -r /root/.setup/requirements.txt
RUN python3 /root/.setup/setup.py --disable-ssh

# installing notebook tqdm for jupyter
RUN conda install jupyterlab -y
RUN conda install -c conda-forge ipywidgets -y
RUN conda upgrade -c conda-forge jupyterlab -y
RUN conda install nodejs -y
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager -y

# installing project requirements.txt
COPY requirements.txt /root/requirements.txt
RUN conda install --file /root/requirements.txt
RUN jupyter lab build

CMD service ssh restart && `which zsh`