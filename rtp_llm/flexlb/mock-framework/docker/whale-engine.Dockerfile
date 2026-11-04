FROM hub.docker.alibaba-inc.com/aone-base-global/flexlb_base:20260519160456544448

ENV APP_NAME=mock-engine
WORKDIR /home/admin/mock-engine
COPY --chown=admin:admin mock-engine.jar build-manifest.json ./
COPY --chown=admin:admin whale_start.sh /home/admin/whale_start.sh
USER admin
HEALTHCHECK NONE
ENTRYPOINT ["/bin/bash", "/home/admin/whale_start.sh"]
CMD []
