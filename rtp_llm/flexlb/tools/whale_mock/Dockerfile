ARG FRONTEND_IMAGE
FROM ${FRONTEND_IMAGE} AS python
FROM hub.docker.alibaba-inc.com/aone-base-global/flexlb_base:20260519160456544448
ARG BUILD_SOURCE_SHA
ENV BUILD_SOURCE_SHA=${BUILD_SOURCE_SHA}
USER root
COPY --from=python /opt/conda310/ /opt/conda310/
COPY tools /opt/whale-mock/tools
COPY jars /opt/whale-mock/tools/whale_mock/jars
COPY tools/whale_mock/start.sh /home/admin/whale_start.sh
RUN chmod +x /home/admin/whale_start.sh && java -version && /opt/conda310/bin/python3 -c 'import yaml'
ENTRYPOINT ["/bin/bash", "/home/admin/whale_start.sh"]
