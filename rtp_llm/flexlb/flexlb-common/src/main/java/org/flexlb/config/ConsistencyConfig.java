package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = NoConsistencyConfig.class, name = "NONE"),
        @JsonSubTypes.Type(value = ZookeeperConsistencyConfig.class, name = "ZOOKEEPER")
})
public sealed interface ConsistencyConfig
        permits NoConsistencyConfig, ZookeeperConsistencyConfig {
}
