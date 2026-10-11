package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/30
 */
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
