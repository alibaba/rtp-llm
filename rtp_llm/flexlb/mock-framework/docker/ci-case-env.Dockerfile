# Build from the public source checkout. This is a Case test environment, not a Whale image.
ARG JAVA_IMAGE=eclipse-temurin:21-jdk-jammy
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm
ARG PROMETHEUS_IMAGE=prom/prometheus:v3.5.0
FROM ghcr.io/astral-sh/uv:0.9.5 AS uv
FROM ${PROMETHEUS_IMAGE} AS prometheus
FROM ${JAVA_IMAGE} AS java
FROM ${PYTHON_IMAGE} AS tools
ARG DEBIAN_MIRROR=http://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=java /opt/java/openjdk /opt/java/openjdk
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=prometheus /bin/prometheus /bin/promtool /usr/local/bin/
ENV JAVA_HOME=/opt/java/openjdk PATH=/opt/java/openjdk/bin:$PATH \
    UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=copy UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never PROMETHEUS_BIN=/usr/local/bin/prometheus

# Warm dependencies, but never ship compiled Mock code in the test environment.
FROM tools AS dependencies
COPY rtp_llm/flexlb /opt/mock-source/rtp_llm/flexlb
COPY rtp_llm/cpp/model_rpc/proto /opt/mock-source/rtp_llm/cpp/model_rpc/proto
RUN uv sync --project /opt/mock-source/rtp_llm/flexlb/mock-framework --locked \
    && /opt/mock-source/rtp_llm/flexlb/mock-framework/scripts/build-java.sh -DskipTests '-P!internal,!internal-monitoring,!internal-discovery'

FROM tools
COPY --from=dependencies /root/.m2 /root/.m2
COPY --from=dependencies /opt/uv-cache /opt/uv-cache
WORKDIR /workspace
CMD ["bash"]
