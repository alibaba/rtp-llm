ARG JAVA_BASE_IMAGE=eclipse-temurin:21-jre
FROM ${JAVA_BASE_IMAGE}
WORKDIR /opt/flexlb
COPY mock-engine.jar /opt/flexlb/mock-engine.jar
COPY start.sh /opt/flexlb/start.sh
COPY mode_defaults.sh /opt/flexlb/mode_defaults.sh
ENTRYPOINT ["/bin/sh", "/opt/flexlb/start.sh"]
