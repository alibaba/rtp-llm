ARG BASE_IMAGE=hub.docker.alibaba-inc.com/isearch/rtp_llm_root_base:2026_04_28_16_56_6fcb8f8
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG BUILD_SOURCE_SHA
ENV BUILD_SOURCE_SHA=${BUILD_SOURCE_SHA} \
    MOCK_RUNTIME_BASE_IMAGE=${BASE_IMAGE} \
    JAVA_HOME=/opt/taobao/java \
    PATH=/opt/taobao/java/bin:${PATH}
USER root
COPY tools /opt/whale-mock/tools
COPY jars /opt/whale-mock/tools/whale_mock/jars
COPY tools/whale_mock/start.sh /home/admin/whale_start.sh
RUN /opt/conda310/bin/pip install --no-cache-dir PyYAML==6.0.2 \
      -i https://artlab.alibaba-inc.com/1/PYPI/simple/ && \
    chmod +x /home/admin/whale_start.sh && \
    test -s /opt/whale-mock/tools/whale_mock/jars/master.jar && \
    test -s /opt/whale-mock/tools/whale_mock/jars/mock.jar && \
    test -s /opt/whale-mock/tools/whale_mock/jars/legacy-master.jar && \
    "$JAVA_HOME/bin/java" -version && \
    /opt/conda310/bin/python3 -c 'import sys, yaml; sys.path.insert(0, "/opt/whale-mock/tools/whale_mock"); import bundle'
ENTRYPOINT ["/bin/bash", "/home/admin/whale_start.sh"]
