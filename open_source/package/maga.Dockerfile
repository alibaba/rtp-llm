ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG WHL_FILE
ADD $WHL_FILE /tmp/$WHL_FILE
RUN /opt/conda310/bin/pip install /tmp/$WHL_FILE \
    -i https://artifacts.antgroup-inc.cn/simple/ \
    --extra-index-url=https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url=https://download.pytorch.org/whl/cu126 \
    && rm /tmp/$WHL_FILE

ARG START_FILE
ADD $START_FILE /usr/bin/maga_start.sh
