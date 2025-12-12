# All load statements must be at the top
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")




# For backward compatibility, keep the original function name
def flexlb_init(name = "flexlb_maven"):
    """Initialize all FlexLB dependencies (for backward compatibility)."""
    return {
        "name": name,
        "artifacts": [
            # Spring Boot dependencies (exact versions from pom.xml)
            "org.springframework.boot:spring-boot-starter:2.7.18",
            "org.springframework.boot:spring-boot:2.7.18",
            "org.springframework.boot:spring-boot-configuration-processor:2.7.18",
            "org.springframework.boot:spring-boot-starter-test:2.7.18",
            "org.springframework.boot:spring-boot-starter-actuator:2.7.18",
            "org.springframework.boot:spring-boot-starter-webflux:2.7.18",
            "org.springframework.boot:spring-boot-autoconfigure:2.7.18",

            # Spring Framework (explicit version 5.3.39)
            "org.springframework:spring-framework-bom:5.3.39",
            "org.springframework:spring-context:5.3.39",
            "org.springframework:spring-beans:5.3.39",
            "org.springframework:spring-context-support:5.3.39",
            "org.springframework:spring-core:5.3.39",

            # Jackson (managed by Spring Boot)
            "com.fasterxml.jackson.core:jackson-databind:2.15.4",
            "com.fasterxml.jackson.core:jackson-core:2.15.4",
            "com.fasterxml.jackson.core:jackson-annotations:2.15.4",
            "com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.15.4",

            # Apache Commons
            "org.apache.commons:commons-lang3:3.12.0",
            "org.apache.commons:commons-collections4:4.4",

            # Micrometer (exact version 1.10.13)
            "io.micrometer:micrometer-bom:1.10.13",
            "io.micrometer:micrometer-registry-prometheus:1.10.13",
            "io.micrometer:micrometer-core:1.10.13",

            # Reactor (exact version 2024.0.10)
            "io.projectreactor:reactor-bom:2024.0.10",
            "io.projectreactor:reactor-core:3.6.11",
            "io.projectreactor:reactor-test:3.6.11",

            # Netty (exact version 4.1.127.Final)
            "io.netty:netty-bom:4.1.127.Final",
            "io.netty:netty-transport:4.1.127.Final",
            "io.netty:netty-common:4.1.127.Final",
            "io.netty:netty-buffer:4.1.127.Final",
            "io.netty:netty-codec:4.1.127.Final",
            "io.netty:netty-codec-http:4.1.127.Final",
            "io.netty:netty-handler:4.1.127.Final",
            "io.netty:netty-codec-http2:4.1.127.Final",
            "io.netty:netty-transport-native-epoll:4.1.127.Final",
            "io.netty:netty-transport-native-unix-common:4.1.127.Final",
            "io.netty:netty-resolver:4.1.127.Final",

            # Caffeine cache (exact version 2.9.3)
            "com.github.ben-manes.caffeine:caffeine:2.9.3",

            # Guava (exact version 33.4.8-jre)
            "com.google.guava:guava:33.4.8-jre",

            # Lombok
            "org.projectlombok:lombok:1.18.30",

            # Byte Buddy (exact version 1.17.6)
            "net.bytebuddy:byte-buddy-parent:1.17.6",

            # Mockito (exact version 5.20.0)
            "org.mockito:mockito-bom:5.20.0",
            "org.mockito:mockito-core:5.20.0",

            # JUnit 5 (Spring Boot managed)
            "org.junit.jupiter:junit-jupiter-api:5.9.3",
            "org.junit.jupiter:junit-jupiter-params:5.9.3",
            "org.junit.jupiter:junit-jupiter-engine:5.9.3",
            "org.junit.platform:junit-platform-console:1.9.3",

            # OpenTelemetry (exact version 1.51.0)
            "io.opentelemetry:opentelemetry-bom:1.51.0",
            "io.opentelemetry:opentelemetry-api:1.51.0",
            "io.opentelemetry:opentelemetry-context:1.51.0",
            "io.opentelemetry:opentelemetry-sdk:1.51.0",
            "io.opentelemetry:opentelemetry-sdk-common:1.51.0",
            "io.opentelemetry:opentelemetry-sdk-trace:1.51.0",
            "io.opentelemetry:opentelemetry-exporter-otlp:1.51.0",

            # SLF4J (Spring Boot managed)
            "org.slf4j:slf4j-api:2.0.13",

            # ZooKeeper and Curator (exact version 5.4.0)
            "org.apache.curator:curator-framework:5.4.0",
            "org.apache.curator:curator-client:5.4.0",
            "org.apache.curator:curator-recipes:5.4.0",
            "org.apache.zookeeper:zookeeper:3.6.3",

            # JSR-250 annotations (Spring Boot managed)
            "javax.annotation:javax.annotation-api:1.3.2",
            "jakarta.annotation:jakarta.annotation-api:1.3.5",

            # gRPC dependencies (exact version 1.65.0)
            "io.grpc:grpc-bom:1.65.0",
            "io.grpc:grpc-all:1.65.0",
            "io.grpc:grpc-api:1.65.0",
            "io.grpc:grpc-context:1.65.0",
            "io.grpc:grpc-core:1.65.0",
            "io.grpc:grpc-services:1.65.0",
            "io.grpc:grpc-stub:1.65.0",
            "io.grpc:grpc-protobuf:1.65.0",
            "io.grpc:grpc-netty:1.65.0",
            "io.grpc:grpc-census:1.65.0",
            "io.grpc:grpc-testing:1.65.0",

            # Protobuf dependencies (exact version 3.25.1)
            "com.google.protobuf:protobuf-java:3.25.1",
            "com.google.protobuf:protobuf-java-util:3.25.1",
        ],
        "repositories": [
            "https://maven.aliyun.com/repository/central/",
            "https://repo1.maven.org/maven2",
            "https://repo.maven.apache.org/maven2",
        ],
    }