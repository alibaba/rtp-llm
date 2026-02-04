package org.flexlb.dao.route;

import lombok.Getter;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import java.util.HashMap;
import java.util.Map;

import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
import static org.flexlb.enums.LoadBalanceStrategyEnum.SHORTEST_TTFT;
import static org.flexlb.enums.LoadBalanceStrategyEnum.WEIGHTED_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.WAIT_TIME;

@Getter
public enum RoleType {
    PDFUSION("RoleType.PDFUSION", "PD融合", SHORTEST_TTFT, WAIT_TIME),
    PREFILL("RoleType.PREFILL", "预填充", SHORTEST_TTFT, WAIT_TIME),
    DECODE("RoleType.DECODE", "解码", WEIGHTED_CACHE, REMAINING_KV_CACHE),
    VIT("RoleType.VIT", "Vision Transformer", RANDOM, WAIT_TIME);

    private final String code;
    private final String description;
    private final LoadBalanceStrategyEnum defaultStrategy;                // 默认负载均衡策略
    private final ResourceMeasureIndicatorEnum resourceMeasureIndicator;  // 资源度量指标

    private static final Map<String, RoleType> CODE_MAP = new HashMap<>();

    static {
        for (RoleType roleType : RoleType.values()) {
            CODE_MAP.put(roleType.code, roleType);
        }
    }

    RoleType(String code, String description, LoadBalanceStrategyEnum defaultStrategy,
             ResourceMeasureIndicatorEnum resourceMeasureIndicator) {
        this.code = code;
        this.description = description;
        this.defaultStrategy = defaultStrategy;
        this.resourceMeasureIndicator = resourceMeasureIndicator;
    }

    /**
     * 检查字符串是否匹配当前角色类型
     */
    public boolean matches(String code) {
        return this.code.equals(code);
    }

    public static RoleType getBy(String code) {
        return CODE_MAP.get(code);
    }

    /**
     * 获取该角色类型的负载均衡策略
     *
     * @param loadBalanceStrategyByConfig 配置的策略
     * @param decodeLoadBalanceStrategy DECODE角色的策略配置
     * @param vitLoadBalanceStrategy VIT角色的策略配置
     * @return 该角色应使用的负载均衡策略
     */
    public LoadBalanceStrategyEnum getStrategy(LoadBalanceStrategyEnum loadBalanceStrategyByConfig,
                                                LoadBalanceStrategyEnum decodeLoadBalanceStrategy,
                                                LoadBalanceStrategyEnum vitLoadBalanceStrategy) {
        // DECODE 使用配置的 decodeLoadBalanceStrategy
        if (this == DECODE) {
            return decodeLoadBalanceStrategy != null ? decodeLoadBalanceStrategy : WEIGHTED_CACHE;
        }
        // VIT 使用配置的 vitLoadBalanceStrategy
        if (this == VIT) {
            return vitLoadBalanceStrategy != null ? vitLoadBalanceStrategy : this.defaultStrategy;
        }
        // 其他角色使用配置的策略
        return loadBalanceStrategyByConfig != null ? loadBalanceStrategyByConfig : this.defaultStrategy;
    }

    /**
     * 根据角色类型获取对应的错误类型.
     *
     * @return 对应的错误类型
     */
    public StrategyErrorType getErrorType() {
        return switch (this) {
            case PREFILL -> StrategyErrorType.NO_PREFILL_WORKER;
            case DECODE -> StrategyErrorType.NO_DECODE_WORKER;
            case PDFUSION -> StrategyErrorType.NO_PDFUSION_WORKER;
            case VIT -> StrategyErrorType.NO_VIT_WORKER;
        };
    }
}
