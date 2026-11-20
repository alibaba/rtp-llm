package org.flexlb.util.network;

import lombok.Getter;

/**
 * Identifies the backend transport protocol used by an endpoint.
 */
@Getter
public enum BackendServiceProtocolEnum {
    HTTP("http"),

    HTTPS("https"),

    WS("ws"),

    WSS("wss"),

    GRPC("grpc");

    private final String name;

    BackendServiceProtocolEnum(String name) {
        this.name = name;
    }
}
