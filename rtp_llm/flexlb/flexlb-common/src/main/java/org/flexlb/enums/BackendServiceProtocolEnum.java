package org.flexlb.enums;

import lombok.Getter;

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
