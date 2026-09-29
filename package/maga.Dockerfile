ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ARG WHL_FILE
ARG EXPECTED_CUDA_MAJOR
ARG REQUIRE_COMPUTE_OPS=1
ARG EXPECT_FLASHINFER_RUNTIME_LIBS=0
ARG PYTORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126
ADD $WHL_FILE /tmp/$WHL_FILE
RUN /opt/conda310/bin/pip install /tmp/$WHL_FILE \
    -i https://artifacts.antgroup-inc.cn/simple/ \
    --extra-index-url=https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url=${PYTORCH_WHEEL_INDEX} \
    && rm /tmp/$WHL_FILE

# Reject CUDA 12-linked RTP-LLM binaries at the CUDA 13 packaging boundary.
RUN if [ "${EXPECTED_CUDA_MAJOR:-}" = "13" ]; then \
        if ! command -v readelf >/dev/null 2>&1; then \
            echo "ERROR: readelf is required for CUDA runtime validation" >&2; \
            exit 1; \
        fi; \
        if ! BAD_CUDA12_ELFS="$(find /opt/conda310/lib/python3.10/site-packages/rtp_llm \
            -type f \( -name '*.so' -o -name '*.so.*' \) \
            -exec sh -c 'for elf do \
                if ! dynamic_section=$(readelf -d "$elf" 2>&1); then \
                    echo "ERROR: readelf failed for $elf: $dynamic_section" >&2; \
                    exit 1; \
                fi; \
                if printf "%s\n" "$dynamic_section" | grep -Eq "NEEDED.*lib(cudart|cupti)\.so\.12"; then \
                    printf "%s\n" "$elf"; \
                fi; \
            done' sh {} +)"; then \
            echo "ERROR: failed to inspect RTP-LLM ELF dependencies" >&2; \
            exit 1; \
        fi; \
        if [ -n "$BAD_CUDA12_ELFS" ]; then \
            echo "ERROR: CUDA 13 image contains RTP-LLM ELF files linked to CUDA 12:" >&2; \
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

ARG START_FILE
ADD $START_FILE /usr/bin/maga_start.sh
ADD dash_sc_start.sh /usr/bin/dash_sc_start.sh
RUN chmod 0755 /usr/bin/dash_sc_start.sh
