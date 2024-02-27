ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ADD requirements_base.txt /opt/conda310/requirements_base.txt

ARG REQUIREMENT_FILE
ADD $REQUIREMENT_FILE /opt/conda310/requirements_gpu.txt
RUN pip3 install -r /opt/conda310/requirements_gpu.txt -i https://artifacts.antgroup-inc.cn/simple/

ARG WHL_FILE
ADD $WHL_FILE /tmp/$WHL_FILE
RUN pip3 install /tmp/$WHL_FILE -i https://artifacts.antgroup-inc.cn/simple/ && \
    rm /tmp/$WHL_FILE
