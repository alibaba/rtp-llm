package org.flexlb.domain.balance;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.LogLevel;

/**
 * @author zjw
 * description:
 * date: 2025/3/10
 */
@Getter
@Setter
public class WhaleMasterConfig {

    /**
     * 负载均衡策略
     */
    private LoadBalanceStrategyEnum loadBalanceStrategy = LoadBalanceStrategyEnum.SHORTEST_TTFT;

    private boolean enableGrpcEngineStatus = true;
    private boolean enableGrpcCacheStatus = true;

    @Getter
    @Setter
    private static LogLevel logLevel;

}
