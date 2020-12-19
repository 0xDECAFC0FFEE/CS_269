FROM debian:stable-20201117-slim
RUN apt update
RUN apt install apt-utils build-essential wget tmux curl git -y
RUN apt install python3-dev python3-pip zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev -y

# installing anaconda python 3.6.6
# RUN cd / && wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh && echo "\nyes\n\n" | /bin/bash Berryconda3-2.0.0-Linux-armv7l.sh
# ENV PATH="/root/berryconda3/bin:"$PATH
# RUN conda install python=3.6 -y
# RUN pip install --upgrade pip

# RUN source activate base

# install pytorch dependencies
# RUN apt-get install python3-numpy
# RUN conda install pyyaml setuptools six requests -y
#   conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
# RUN apt install cmake -y

# download pytorch source code
# RUN cd / && git clone --recursive https://github.com/pytorch/pytorch
# RUN cd /pytorch && git submodule sync && git submodule update --init --recursive


RUN apt install libopenblas-dev libblas-dev m4 cmake cython -y
# install openblas - eigen should work as well but is slower to install and ive been sitting here for hours
RUN apt-get install gfortran -y
# RUN cd / && git clone https://github.com/xianyi/OpenBLAS.git && cd OpenBLAS && make TARGET=ARMV7 && make install && cd / && rm -rf OpenBLAS
RUN apt install python3-yaml python3-pillow -y
RUN pip3 install setuptools wheel numpy
COPY compiled_pytorch_wheel_python_3_7_armv7 compiled_pytorch_wheel_python_3_7_armv7
RUN pip3 install compiled_pytorch_wheel_python_3_7_armv7/torch-1.4.0a0-cp37-cp37m-linux_armv7l.whl compiled_pytorch_wheel_python_3_7_armv7/torchvision-0.5.0a0-cp37-cp37m-linux_armv7l.whl && rm -r compiled_pytorch_wheel_python_3_7_armv7

# eigen should be faster than openblas but takes wayy longer to install
# RUN apt-get install libeigen3-dev -y

# RUN cd / && mkdir eigen-3.3.8
# RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.zip -O /eigen-3.3.8/eigen.zip
# RUN apt install unzip -y
# RUN unzip /eigen-3.3.8/eigen.zip -d /
# RUN mkdir /eigen-3.3.8/build_dir
# RUN cd /eigen-3.3.8/build_dir && cmake -DCMAKE_BUILD_TYPE=Release /eigen-3.3.8 && make blas lapack install

# RUN /root/berryconda3/bin/python3 -m pip install numpy ninja cffi typing_extensions future dataclasses
# RUN conda install opencv -y
# RUN pip3 install cmake

# RUN export USE_NUMPY=1

# mkl is going to be rough... specific to intel cpus for highly optimized vectorized operations. using openblas instead.

# install pytorch
# RUN cd /pytorch && USE_CUDA=0 USE_MKLDNN=0 USE_CUDNN=0 USE_MKL=0 USE_LAPACK=1 BLAS=OpenBLAS CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"/root/berryconda3/bin/../"} python3 setup.py install

# # set up ssh
# RUN apt-get -y install openssh-server && echo "root ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && passwd -d `whoami` && echo "Port 7722\nPermitEmptyPasswords yes\nX11Forwarding yes\nPrintMotd no\nAcceptEnv LANG LC_*\nSubsystem       sftp    /usr/lib/openssh/sftp-server\nPasswordAuthentication yes\nPermitRootLogin yes" > /etc/ssh/sshd_config
# EXPOSE 7722

# install zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)" && chsh -s `which zsh`

# install lucas's env
ADD https://api.github.com/repos/0xDECAFC0FFEE/.setup/git/refs/ version.json
RUN git clone https://github.com/0xDECAFC0FFEE/.setup.git /root/.setup --recursive && python3 /root/.setup/setup.py --disable-ssh

# skipping jupyter install
# RUN conda install jupyterlab -y
# RUN conda install -c conda-forge ipywidgets -y
# RUN conda upgrade -c conda-forge jupyterlab -y
# RUN conda install nodejs -y
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager -y

RUN pip3 install gdown tqdm pillow rigl-torch
RUN apt install unzip

# installing project requirements.txt
# RUN conda config --append channels conda-forge
# COPY requirements_slim.txt /root/requirements.txt
# RUN conda install --file /root/requirements.txt

# CMD ./start_jupyter_tensorboard_ssh.sh && `which zsh`

CMD `which zsh`