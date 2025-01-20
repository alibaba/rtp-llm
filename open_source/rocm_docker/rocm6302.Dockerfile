ARG BASE_OS_IMAGE
FROM $BASE_OS_IMAGE

MAINTAINER wangyin.yx

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    groupadd sdev && touch /root/.bashrc

RUN dnf install -y \
        unzip wget which findutils rsync tar \
        gcc gcc-c++ libstdc++-static gdb coreutils \
        binutils bash glibc-devel libdb glibc glibc-langpack-en bison lld \
        emacs-nox git git-lfs nfs-utils java-17-openjdk-headless \
        gcc-toolset-12 gcc-toolset-12-gcc-c++ libappstream-glib* \
        net-tools http://yum.tbsite.net/taobao/7/noarch/current/t-midware-vipserver-dnsclient/t-midware-vipserver-dnsclient-1.1.11-20240313120706.alios7.noarch.rpm \
        https://mirrors.aliyun.com/docker-ce/linux/centos/8/x86_64/stable/Packages/docker-ce-cli-26.1.3-1.el8.x86_64.rpm

RUN git config --system core.hooksPath .githooks && \
    git lfs install

ENV PATH /opt/rh/gcc-toolset-12/root/usr/bin:$PATH:/opt/conda310/bin:/opt/rocm/bin
ENV LD_LIBRARY_PATH /opt/rh/gcc-toolset-12/root/usr/lib64:$LD_LIBRARY_PATH:/lib64:/opt/conda310/lib/

ARG CONDA_URL
RUN wget $CONDA_URL -O /tmp/conda.sh && \
    sh /tmp/conda.sh -b -p /opt/conda310/ && \
    rm /tmp/conda.sh -f

ADD deps/requirements_rocm.txt /deps/requirements_rocm.txt
RUN /opt/conda310/bin/python3 -m pip install -r /deps/requirements_rocm.txt -i https://rtp-pypi-mirrors.alibaba-inc.com/root/pypi/+simple/

ARG BAZELISK_URL
RUN wget -q $BAZELISK_URL -O /usr/local/bin/bazelisk && chmod a+x /usr/local/bin/bazelisk
