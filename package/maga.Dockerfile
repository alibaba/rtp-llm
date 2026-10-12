ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG WHL_FILE
ADD $WHL_FILE /tmp/$WHL_FILE
COPY cuda13_uv_overrides.txt /tmp/cuda13_uv_overrides.txt
RUN /opt/conda310/bin/pip install uv==0.11.1 && \
    /opt/conda310/bin/uv pip install --python=/opt/conda310/bin/python /tmp/$WHL_FILE \
    --overrides /tmp/cuda13_uv_overrides.txt \
    -i https://artifacts.antgroup-inc.cn/simple/ \
    --extra-index-url=https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url=https://download.pytorch.org/whl/cu126 \
    && rm /tmp/$WHL_FILE /tmp/cuda13_uv_overrides.txt

# pip installation does not inherit Bazel wheel patches.
COPY patches/ /tmp/cuda13-patches/
COPY apply_flashinfer_patch.sh apply_tilelang_patch.sh validate_cuda13_runtime.py /tmp/
RUN CUDA_MAJOR="$(/opt/conda310/bin/python -c 'import torch; print((torch.version.cuda or "").split(".")[0])')" && \
    if [ "$CUDA_MAJOR" = "13" ]; then \
        (command -v patch >/dev/null || yum install -y patch) && \
        bash /tmp/apply_flashinfer_patch.sh /tmp/cuda13-patches && \
        EXPECTED_CUDA_MAJOR=13 bash /tmp/apply_tilelang_patch.sh /tmp/cuda13-patches && \
        INCOMPATIBLE_CACHES="$(/opt/conda310/bin/python /tmp/validate_cuda13_runtime.py --incompatible-caches)" && \
        if [ -n "$INCOMPATIBLE_CACHES" ]; then \
            /opt/conda310/bin/uv pip uninstall --python=/opt/conda310/bin/python $INCOMPATIBLE_CACHES; \
        fi && \
        /opt/conda310/bin/python /tmp/validate_cuda13_runtime.py; \
    fi && rm -r /tmp/cuda13-patches /tmp/apply_flashinfer_patch.sh \
        /tmp/apply_tilelang_patch.sh /tmp/validate_cuda13_runtime.py

ARG START_FILE
ADD $START_FILE /usr/bin/maga_start.sh
ADD dash_sc_start.sh /usr/bin/dash_sc_start.sh
RUN chmod 0755 /usr/bin/dash_sc_start.sh
