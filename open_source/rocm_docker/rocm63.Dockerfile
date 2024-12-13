ARG BASE_OS_IMAGE
FROM $BASE_OS_IMAGE

MAINTAINER wangyin.yx

ARG AMD_BKC_URL

RUN dnf install -y \
        unzip wget which findutils rsync tar \
        gcc gcc-c++ libstdc++-static gdb coreutils \
        binutils bash glibc-devel libdb glibc glibc-langpack-en bison lld \
        emacs-nox git git-lfs nfs-utils java-17-openjdk-devel \
        gcc-toolset-12 gcc-toolset-12-gcc-c++ libappstream-glib* \
        https://mirrors.aliyun.com/docker-ce/linux/centos/8/x86_64/stable/Packages/docker-ce-cli-26.1.3-1.el8.x86_64.rpm

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    groupadd sdev && touch /root/.bashrc

RUN wget $AMD_BKC_URL -O /tmp/bkc.tar.gz && \
    mkdir -p /tmp/bkc && \
    tar -xzvf /tmp/bkc.tar.gz -C /tmp/bkc && \
    cd /tmp/bkc/ali_8u && \
    bash install_ch.sh && \
    cd ./rpmpackages && \
    yum install -y *rand-* *blas* hipfft-* \
        miopen-hip-* hipsolver-* hipsparse* \
        composablekernel-devel-1.1.0.6030001-16.el8.x86_64.rpm \
        rocthrust-devel-3.0.1.6030001-16.el8.x86_64.rpm \
        hipcub-devel-3.2.0.6030001-16.el8.x86_64.rpm && \
    yum config-manager --disable local-rocm && \
    rm -rf /tmp/bkc /tmp/bkc.tar.gz

RUN git config --system core.hooksPath .githooks && \
    git lfs install

ENV PATH /opt/rh/gcc-toolset-12/root/usr/bin:$PATH:/opt/conda310/bin:/opt/rocm/bin
ENV LD_LIBRARY_PATH /opt/rh/gcc-toolset-12/root/usr/lib64:$LD_LIBRARY_PATH:/lib64:/opt/conda310/lib/

ARG CONDA_URL
RUN wget $CONDA_URL -O /tmp/conda.sh && \
    sh /tmp/conda.sh -b -p /opt/conda310/ && \
    rm /tmp/conda.sh -f

ARG BAZELISK_URL
RUN wget -q $BAZELISK_URL -O /usr/local/bin/bazelisk && chmod a+x /usr/local/bin/bazelisk
