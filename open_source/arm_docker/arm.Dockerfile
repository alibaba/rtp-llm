ARG BASE_OS_IMAGE
FROM $BASE_OS_IMAGE

MAINTAINER wangyin.yx

RUN echo "%sdev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    groupadd sdev

RUN dnf install -y \
        unzip wget which findutils rsync tar \
        gcc gcc-c++ libstdc++-static gdb coreutils \
        binutils bash glibc-devel libdb glibc glibc-langpack-en bison lld \
        emacs-nox git git-lfs openblas-devel


ARG CONDA_URL
RUN wget $CONDA_URL -O /tmp/conda.sh && \
    sh /tmp/conda.sh -b -p /opt/conda310/ && \
    rm /tmp/conda.sh -f

ARG BAZEL_URL
RUN wget $BAZEL_URL -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

RUN git config --system core.hooksPath .githooks && \
    git lfs install

ARG PYPI_URL
ADD deps /tmp/deps
RUN /opt/conda310/bin/pip install -r /tmp/deps/requirements_base.txt -i $PYPI_URL && \
    rm -rf /tmp/deps && /opt/conda310/bin/pip cache purge
