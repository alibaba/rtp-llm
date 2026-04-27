ARG FROM_IMAGE
FROM $FROM_IMAGE

RUN yum install git-2.19.1 git-lfs -bcurrent -y && \
    git config --system core.hooksPath .githooks && \
    git lfs install

Run echo "add docker client for Docker-out-Docker" && yum install -y docker-ce-cli-20.10.6-3.el7.x86_64 -b current -y

# install alibaba clang13/llvm13
RUN yum -y install yum-plugin-ovl && yum clean all && \
    yum -y install llvm13 llvm13-devel llvm13-libs llvm13-static --enablerepo taobao.7.x86_64.current -b current
RUN yum clean all && yum -y install clang13-libs clang13 clang13-devel clang13-tools-extra compiler-rt13 --enablerepo taobao.7.x86_64.current -b current
RUN yum clean all && yum -y install lld13 --enablerepo taobao.7.x86_64.current -b current

# https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
ARG BAZELISK_URL=http://search-ad.oss-cn-hangzhou-zmf-internal.aliyuncs.com/aios/bazelisk-1.20.0
RUN wget -q $BAZELISK_URL -O /usr/local/bin/bazelisk && chmod a+x /usr/local/bin/bazelisk

ENV BAZELISK_HOME=/opt/bazelisk
ENV BAZELISK_BASE_URL=https://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/bazel_binary
RUN USE_BAZEL_VERSION=6.4.0 bazelisk # prefetch bazel

RUN rpm -i http://mirrors.aliyun.com/epel/7/x86_64/Packages/p/patchelf-0.12-1.el7.x86_64.rpm # for bazel rules

RUN echo "%sdev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    groupadd sdev

# for pre-commit
RUN /opt/conda310/bin/python3 -m pip install pre-commit && ln -s /opt/conda310/bin/pre-commit /usr/local/bin/pre-commit