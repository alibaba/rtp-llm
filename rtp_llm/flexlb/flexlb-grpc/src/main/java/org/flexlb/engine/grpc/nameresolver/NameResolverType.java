package org.flexlb.engine.grpc.nameresolver;

import lombok.Getter;

/**
 * @author zjw
 * description:
 * date: 2025/4/17
 */
@Getter
public enum NameResolverType {

    VIPSERVER("vipserver"),
    STATIC_HOSTS("whaledirect");

    private final String schema;

    NameResolverType(String schema) {
        this.schema = schema;
    }
}
