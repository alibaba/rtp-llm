package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // 随机分配

    SPECIFIED_IP_PORT("SpecifiedIpPort"), // 指定机器ip:port

    SPECIFIED_IP_PORT_LIST("SpecifiedIpPortList"), // 指定机器ip:port列表

    SERVICE_DISCOVERY("ServiceDiscovery"), // 服务发现

    SHORTEST_TTFT("ShortestTTFT"),  // 最短TTFT

    LOWEST_CACHE_USED("LowestCacheUsed")  // 最低缓存使用策略

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
