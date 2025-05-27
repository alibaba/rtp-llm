# https://docker.alibaba-inc.com/#/dockerImage/3124085/detail

ARG FROM_IMAGE
FROM $FROM_IMAGE

MAINTAINER liukan.lk

RUN --mount=type=secret,id=id --mount=type=secret,id=secret cd /tmp/ && \
    osscmd multiget --thread_num=30 oss://search-ad/pkg/cuda_12.6.3_560.35.05_linux.run 1.run --id=$(cat /run/secrets/id) --key=$(cat /run/secrets/secret) --host=oss-cn-hangzhou-zmf-internal.aliyuncs.com && \
    sh 1.run --silent --toolkit && \
    rm -f 1.run

RUN --mount=type=secret,id=id --mount=type=secret,id=secret cd / && \
    osscmd multiget --thread_num=30 oss://search-ad/pkg/cuda-compat-12-6-560.35.05-1.el8.x86_64.rpm 1.rpm --id=$(cat /run/secrets/id) --key=$(cat /run/secrets/secret) --host=oss-cn-hangzhou-zmf-internal.aliyuncs.com && \
    rpm2cpio 1.rpm | cpio -div && \
    rm -f 1.rpm

RUN --mount=type=secret,id=id --mount=type=secret,id=secret cd /tmp/ && \
    osscmd multiget --thread_num=30 oss://search-ad/pkg/nccl-local-repo-rhel8-2.22.3-cuda12.5-1.0-1.x86_64.rpm 1.rpm --id=$(cat /run/secrets/id) --key=$(cat /run/secrets/secret) --host=oss-cn-hangzhou-zmf-internal.aliyuncs.com && \
    rpm2cpio 1.rpm | cpio -div && \
    rpm2cpio ./var/nccl-local-repo-rhel8-2.22.3-cuda12.5/libnccl-devel-2.22.3-1+cuda12.5.x86_64.rpm | cpio -div && \
    rpm2cpio ./var/nccl-local-repo-rhel8-2.22.3-cuda12.5/libnccl-2.22.3-1+cuda12.5.x86_64.rpm | cpio -div && \
    mv ./usr/include/* /usr/local/cuda/include/ && \
    mv ./usr/lib64/* /usr/local/cuda/lib64/ && \
    rm ./var/ ./etc/ ./usr/ -rf && \
    rm -f 1.rpm

RUN --mount=type=secret,id=id --mount=type=secret,id=secret cd /tmp/ && \
    osscmd multiget --thread_num=30 oss://search-ad/pkg/cudnn-local-repo-rhel8-9.5.1-1.0-1.x86_64.rpm 1.rpm --id=$(cat /run/secrets/id) --key=$(cat /run/secrets/secret) --host=oss-cn-hangzhou-zmf-internal.aliyuncs.com && \
    rpm2cpio 1.rpm | cpio -div && \
    rpm2cpio ./var/cudnn-local-repo-rhel8-9.5.1/libcudnn9-cuda-12-9.5.1.17-1.x86_64.rpm | cpio -div && \
    rpm2cpio ./var/cudnn-local-repo-rhel8-9.5.1/libcudnn9-devel-cuda-12-9.5.1.17-1.x86_64.rpm | cpio -div && \
    mv ./usr/include/* /usr/local/cuda/include/ && \
    mv ./usr/lib64/* /usr/local/cuda/lib64/ && \
    rm ./var/ ./etc/ ./usr/ -rf && \
    rm -f 1.rpm

RUN --mount=type=secret,id=id --mount=type=secret,id=secret cd /tmp/ && \
    osscmd multiget --thread_num=30 oss://search-ad/pkg/cusparselt-local-repo-rhel7-0.7.1-1.0-1.x86_64.rpm 1.rpm --id=$(cat /run/secrets/id) --key=$(cat /run/secrets/secret) --host=oss-cn-hangzhou-zmf-internal.aliyuncs.com && \
    rpm2cpio 1.rpm | cpio -div && \
    rpm2cpio ./var/cusparselt-local-repo-rhel7-0.7.1/libcusparselt0-0.7.1.0-1.x86_64.rpm | cpio -div && \
    rpm2cpio ./var/cusparselt-local-repo-rhel7-0.7.1/libcusparselt-devel-0.7.1.0-1.x86_64.rpm | cpio -div && \
    rm ./usr/lib64/*.a && \
    mv ./usr/include/* /usr/local/cuda/include/ && \
    mv ./usr/lib64/* /usr/local/cuda/lib64/ && \
    rm ./var/ ./etc/ ./usr/ -rf && \
    rm -f 1.rpm

