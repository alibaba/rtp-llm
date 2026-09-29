ENV LD_LIBRARY_PATH=/opt/conda310/lib/python3.10/site-packages/nvidia/nvshmem/lib:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG WHL_FILE
ARG REQUIREMENTS_LOCK_FILE
ARG EXPECTED_CUDA_MAJOR
ARG REQUIRE_COMPUTE_OPS=1
ARG EXPECT_FLASHINFER_RUNTIME_LIBS=0
ARG EXPECT_FAST_HADAMARD=0
ARG EXPECT_FLASH_ATTN_2=0
ADD $WHL_FILE /tmp/$WHL_FILE
ADD $REQUIREMENTS_LOCK_FILE /tmp/runtime-requirements.lock
ADD validate_cuda13_runtime.py /tmp/validate_cuda13_runtime.py
RUN /opt/conda310/bin/pip install uv -i https://mirrors.aliyun.com/pypi/simple/
RUN /opt/conda310/bin/uv pip sync \
        --require-hashes \
        /tmp/runtime-requirements.lock \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --index-strategy unsafe-best-match \
        --python=/opt/conda310/bin/python \
        --verbose && \
    /opt/conda310/bin/python -m pip install --no-deps /tmp/$WHL_FILE && \
    rm /tmp/$WHL_FILE /tmp/runtime-requirements.lock

RUN if [ "${EXPECT_FAST_HADAMARD:-}" = "1" ]; then \
        /opt/conda310/bin/python -c 'import importlib.metadata as m; from fast_hadamard_transform import hadamard_transform; version=m.version("fast-hadamard-transform"); assert version == "1.1.0+e7706fa.cu132.torch2.11.cxx11abitrue", version; assert callable(hadamard_transform); print("validated fast-hadamard-transform", version)'; \
    fi

RUN if [ "${EXPECT_FLASH_ATTN_2:-}" = "1" ]; then \
        /opt/conda310/bin/python -c 'import importlib.metadata as m; import torch; import flash_attn_2_cuda; from flash_attn import flash_attn_func, flash_attn_varlen_func; from flash_attn.bert_padding import pad_input, unpad_input; version=m.version("flash-attn"); assert version == "2.8.3.post1+cu13torch2.11cxx11abitrue.r1", version; assert all(callable(fn) for fn in (flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input)); print("validated flash-attn", version)'; \
    fi

# Reject CUDA 12-linked runtime packages at the CUDA 13 packaging boundary.
RUN if [ "${EXPECTED_CUDA_MAJOR:-}" = "13" ]; then \
        if ! command -v readelf >/dev/null 2>&1; then \
            echo "ERROR: readelf is required for CUDA runtime validation" >&2; \
            exit 1; \
        fi; \
        if ! BAD_CUDA12_ELFS="$(set -e; for root in \
            /opt/conda310/lib/python3.10/site-packages/rtp_llm \
            /opt/conda310/lib/python3.10/site-packages/deep_ep \
            /opt/conda310/lib/python3.10/site-packages/deep_gemm \
            /opt/conda310/lib/python3.10/site-packages/flashinfer \
            /opt/conda310/lib/python3.10/site-packages/rtp_kernel \
            /opt/conda310/lib/python3.10/site-packages/torch \
            /opt/conda310/lib/python3.10/site-packages/nvidia; do \
            [ -d "$root" ] || continue; \
            find -L "$root" -xdev -type f \( -name '*.so' -o -name '*.so.*' \) \
                -exec sh -c 'for elf do \
                    if ! dynamic_section=$(readelf -d "$elf" 2>&1); then \
                        echo "ERROR: readelf failed for $elf: $dynamic_section" >&2; \
                        exit 1; \
                    fi; \
                    if printf "%s\n" "$dynamic_section" | grep -Eq "NEEDED.*lib(cudart|cupti)\.so\.12"; then \
                        printf "%s\n" "$elf"; \
                    fi; \
                done' sh {} +; \
        done)"; then \
            echo "ERROR: failed to inspect runtime ELF dependencies" >&2; \
            exit 1; \
        fi; \
        if [ -n "$BAD_CUDA12_ELFS" ]; then \
            echo "ERROR: CUDA 13 image contains ELF files linked to CUDA 12:" >&2; \
            echo "$BAD_CUDA12_ELFS" >&2; \
            exit 1; \
        fi; \
    fi

# Full runtime wheels must contain the compute library. Frontend-only image jobs
# explicitly set REQUIRE_COMPUTE_OPS=0; no other image may silently skip this check.
RUN RTP_LIB_DIR=/opt/conda310/lib/python3.10/site-packages/rtp_llm/libs; \
    COMPUTE_OPS="${RTP_LIB_DIR}/librtp_compute_ops.so"; \
    case "${REQUIRE_COMPUTE_OPS}" in \
        0) echo "Frontend-only wheel: compute library is not required" ;; \
        1) if [ ! -f "${COMPUTE_OPS}" ]; then \
               echo "ERROR: full runtime wheel is missing ${COMPUTE_OPS}" >&2; \
               exit 1; \
           fi ;; \
        *) echo "ERROR: REQUIRE_COMPUTE_OPS must be 0 or 1" >&2; exit 1 ;; \
    esac

# Production imports the installed wheel without Bazel runfiles, so ensure its
# compute library can resolve every FlashInfer runtime dependency it declares.
RUN if [ "${EXPECT_FLASHINFER_RUNTIME_LIBS:-}" = "1" ]; then \
        RTP_LIB_DIR=/opt/conda310/lib/python3.10/site-packages/rtp_llm/libs; \
        COMPUTE_OPS="${RTP_LIB_DIR}/librtp_compute_ops.so"; \
        if [ ! -f "${COMPUTE_OPS}" ]; then \
            echo "ERROR: FlashInfer validation requires ${COMPUTE_OPS}" >&2; \
            exit 1; \
        fi; \
        if ! command -v ldd >/dev/null 2>&1; then \
            echo "ERROR: ldd is required for FlashInfer runtime validation" >&2; \
            exit 1; \
        fi; \
        if ! LDD_OUTPUT="$(LD_LIBRARY_PATH="${RTP_LIB_DIR}:${LD_LIBRARY_PATH:-}" ldd "${COMPUTE_OPS}" 2>&1)"; then \
            echo "ERROR: ldd failed for ${COMPUTE_OPS}:" >&2; \
            echo "${LDD_OUTPUT}" >&2; \
            exit 1; \
        fi; \
        MISSING_FLASHINFER_DEPS="$(printf '%s\n' "${LDD_OUTPUT}" | grep -E 'libflashinfer_.*=> not found' || true)"; \
        if [ -n "${MISSING_FLASHINFER_DEPS}" ]; then \
            echo "ERROR: CUDA 13 RTP-LLM wheel is missing FlashInfer runtime libraries:" >&2; \
            echo "${MISSING_FLASHINFER_DEPS}" >&2; \
            exit 1; \
        fi; \
    fi

RUN if [ "${EXPECTED_CUDA_MAJOR:-}" = "13" ]; then \
        /opt/conda310/bin/python /tmp/validate_cuda13_runtime.py \
            --expected-cuda-major 13 && \
        rm -f /tmp/validate_cuda13_runtime.py; \
    fi

ARG START_FILE
ADD $START_FILE /usr/bin/maga_start.sh
ADD dash_sc_start.sh /usr/bin/dash_sc_start.sh
RUN chmod 0755 /usr/bin/dash_sc_start.sh
