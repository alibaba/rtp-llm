
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG BAZEL_INSTALL_SH
ADD $BAZEL_INSTALL_SH /tmp/$BAZEL_INSTALL_SH

# Install my-extra-package-1 and my-extra-package-2
RUN apt-get update &&  apt install \
    zip g++ unzip default-jdk -y

RUN chmod +x /tmp/$BAZEL_INSTALL_SH \
    && bash /tmp/$BAZEL_INSTALL_SH
   
RUN if [ -f '/usr/bin/python3.10' ]; \ 
    then \
	echo "python3.10 has exist, so not need to install it"; ls -lrt /usr/bin/python3.10; \
    else \
	echo "python3.10 not exist, so install it"; \
        rm -f -r /opt/conda; \
	apt clean ; apt autoclean; apt update; \
	DEBIAN_FRONTEND=noninteractive apt -y install tzdata; \
	apt -y  install software-properties-common; \
	add-apt-repository ppa:deadsnakes/ppa; \
	apt update; \
	apt -y install python3.10; \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 ; \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 ; \
	update-alternatives --config python3; \
	apt -y install python3.10-dev; \
	apt -y install python3.10-distutils; \
	curl -fSL https://bootstrap.pypa.io/get-pip.py | python3.10; \
    fi

#uninstall protobuf because the version of protobuf is not compatible
RUN apt -y install rpm2cpio \
    && apt -y install cpio \
    && apt install patchelf \
    && pip uninstall -y protobuf

## uninstall transformer-engine because the version of transformer-engine is not compatible
RUN pip3 uninstall -y transformer-engine

RUN apt -y install git git-lfs \
    && git config --system core.hooksPath .githooks \
    && git lfs install

## prepare python && cudnn env 
RUN mkdir -p /opt/conda310/bin/ \
    && ln -s /usr/bin/python3.10 /opt/conda310/bin/python3 \
    && ln -s /usr/bin/python3.10 /opt/conda310/bin/python

RUN if [ -f '/usr/bin/python' ]; \
    then \
	echo "python has exist, so not need to ln it"; ls -lrt /usr/bin/python; \
    else \
	ln -s /opt/conda310/bin/python3 /usr/bin/python; \
    fi
 
RUN ln -s /usr/include/cudnn.h  /usr/local/cuda/include/cudnn.h \
    && ln -s /usr/include/cudnn_version.h  /usr/local/cuda/include/cudnn_version.h \
    && ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so \
    && ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.8
