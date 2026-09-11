FROM eclipse-temurin:21-jre
WORKDIR /opt/flexlb
COPY mock-engine.jar /opt/flexlb/mock-engine.jar
COPY start.sh /opt/flexlb/start.sh
ENTRYPOINT ["/bin/sh", "/opt/flexlb/start.sh"]
