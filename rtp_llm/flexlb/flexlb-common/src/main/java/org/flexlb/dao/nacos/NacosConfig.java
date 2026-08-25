package org.flexlb.dao.nacos;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import static org.flexlb.constant.NacosConfigConstants.DEFAULT_NACOS_GROUP;

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
