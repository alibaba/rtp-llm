package org.flexlb.config.loading.source;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import static org.flexlb.config.loading.source.NacosConfigConstants.DEFAULT_NACOS_GROUP;

/**
 * Resolves the data ID, group and namespace used to subscribe to Nacos configuration.
 */
@Getter
@EqualsAndHashCode
public class NacosConfig {

    private final String serverAddr;
    private final String dataId;
    private String group = DEFAULT_NACOS_GROUP;
    private String namespace = "";

    public NacosConfig(String serverAddr, String dataId, String group, String namespace) {
        this.serverAddr = serverAddr;
        this.dataId = dataId;
        if (group != null) {
            this.group = group;
        }
        if (namespace != null) {
            this.namespace = namespace;
        }
    }
}
