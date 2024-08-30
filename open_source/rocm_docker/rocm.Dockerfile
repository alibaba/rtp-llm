ARG BASE_OS_IMAGE
FROM $BASE_OS_IMAGE

MAINTAINER wangyin.yx

ADD functions /etc/rc.d/init.d/functions

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    groupadd sdev

RUN dnf install -y \
        unzip wget which findutils rsync tar \
        gcc gcc-c++ libstdc++-static gdb coreutils \
        binutils bash glibc-devel libdb glibc glibc-langpack-en bison lld \
        emacs-nox git git-lfs nfs-utils java-11-openjdk-devel docker && \
    rpm --rebuilddb

ARG CONDA_URL
RUN wget $CONDA_URL -O /tmp/conda.sh && \
    sh /tmp/conda.sh -b -p /opt/conda310/ && \
    rm /tmp/conda.sh -f

ARG BAZEL_URL
RUN wget $BAZEL_URL -O /tmp/bazel.sh && sh /tmp/bazel.sh && rm /tmp/bazel.sh -f

RUN git config --system core.hooksPath .githooks && \
    git lfs install

ARG AMD_BKC_URL
RUN wget $AMD_BKC_URL -O /tmp/bkc.tar && \
    mkdir -p /tmp/bkc && \
    tar -xvf /tmp/bkc.tar --strip-components=1 -C /tmp/bkc && \
    cd /tmp/bkc && \
    sed 's/amdgpu-install/amdgpu-install --no-dkms/g' -i install.sh && \
    sh install.sh && \
    rm -rf /tmp/bkc /tmp/bkc.tar && \
    yum config-manager --disable amdgpu-local-rpmpackages && \
    yum config-manager --disable local-rocm

ENV PATH $PATH:/opt/conda310/bin

ARG PYPI_URL
ADD deps /tmp/deps
RUN /opt/conda310/bin/pip install -r /tmp/deps/requirements_rocm.txt -i $PYPI_URL && \
    rm -rf /tmp/deps && pip cache purge

ARG BAZELISK_URL
RUN wget -q $BAZELISK_URL -O /usr/local/bin/bazelisk && chmod a+x /usr/local/bin/bazelisk
