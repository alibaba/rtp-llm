package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class LBConsistencyConfig {

    private boolean needConsistency;
    private MasterElectType masterElectType = MasterElectType.ZOOKEEPER;
    private ZookeeperConfig zookeeperConfig;

    public enum MasterElectType {
        ZOOKEEPER
    }

    @Getter
    @Setter
    public static class ZookeeperConfig {
        private String zkHost;
        private int zkTimeoutMs;
    }
}
