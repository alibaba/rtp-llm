ARG FRONTEND_IMAGE
FROM ${FRONTEND_IMAGE} AS adapter

ARG RTP_PYTHON=/opt/conda310/bin/python3
ARG RTP_CAPTURE_RUNTIME=current
ARG RTP_CAPTURE_PIP_INDEX_URL=https://artlab.alibaba-inc.com/1/PYPI/simple/
USER root
COPY bundle.py capture.py configure.py hook.py layout.py mckidx_writer.py partitioned_index.py prefix_index.py verify.py offline.py /opt/rtp-frontend-capture/
RUN "${RTP_PYTHON}" -m pip install --no-deps --index-url "${RTP_CAPTURE_PIP_INDEX_URL}" --target /opt/rtp-frontend-capture/deps zstandard==0.25.0 \
    && "${RTP_PYTHON}" /opt/rtp-frontend-capture/configure.py http /opt/rtp-frontend-capture/http.json --runtime "${RTP_CAPTURE_RUNTIME}" \
    && "${RTP_PYTHON}" /opt/rtp-frontend-capture/configure.py grpc /opt/rtp-frontend-capture/grpc.json --runtime "${RTP_CAPTURE_RUNTIME}" \
    && "${RTP_PYTHON}" /opt/rtp-frontend-capture/configure.py both /opt/rtp-frontend-capture/both.json --runtime "${RTP_CAPTURE_RUNTIME}" \
    && "${RTP_PYTHON}" /opt/rtp-frontend-capture/capture.py --help

FROM ${FRONTEND_IMAGE}
COPY --from=adapter /opt/rtp-frontend-capture/ /opt/rtp-frontend-capture/
