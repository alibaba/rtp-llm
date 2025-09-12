package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // 随机分配

    ROUND_ROBIN("RoundRobin"),  // 轮询分配

    SPECIFIED_IP_PORT("SpecifiedIpPort"), // 指定机器ip:port

    SPECIFIED_IP_PORT_LIST("SpecifiedIpPortList"), // 指定机器ip:port列表

    HOST_PORT("HostPort"),  // 域名

    VIPSERVER("VipServer"), // Vipserver

    PURE_VIPSERVER("PureVipserver"),  // 仅Vipserver, 无其他负载均衡

    SYNC_BALANCE("VipServer"), // Sync 负载均衡

    PREFILL_BALANCE("PrefillBalance"), // PreFill 负载均衡

    DECODE_BALANCE("DecodeBalance"), // Decode 负载均衡

    WEIGHTED("Weighted"),  // 基于权重分配

    MAX_THROUGHPUT_FIRST("MaxThroughputFirst"),  // 最大吞吐量优先策略

    LOWEST_CONCURRENCY("LowestConcurrency"), // 基于最低并发度分配

    ROUND_ROBIN_LOWEST_CONCURRENCY("LowestConcurrency"), // 基于RoundRobin以及最低并发度分配

    SHORTEST_TTFT("ShortestTTFT"),  // 最短TTFT

    LOWEST_CACHE_USED("LowestCacheUsed")  // 最低缓存使用策略

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
