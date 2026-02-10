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
    PDFUSION("RoleType.PDFUSION", "Prefill-Decode Fusion", SHORTEST_TTFT, WAIT_TIME),
    PREFILL("RoleType.PREFILL", "Prefill", SHORTEST_TTFT, WAIT_TIME),
    DECODE("RoleType.DECODE", "Decode", WEIGHTED_CACHE, REMAINING_KV_CACHE),
    VIT("RoleType.VIT", "Vision Transformer", RANDOM, WAIT_TIME);

    private final String code;
    private final String description;
    private final LoadBalanceStrategyEnum defaultStrategy;                // Default load balancing strategy
    private final ResourceMeasureIndicatorEnum resourceMeasureIndicator;  // Resource measure indicator

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
     * Check if string matches current role type
     */
    public boolean matches(String code) {
        return this.code.equals(code);
    }

    public static RoleType getBy(String code) {
        return CODE_MAP.get(code);
    }

    /**
     * Get load balancing strategy for this role type
     *
     * @param loadBalanceStrategyByConfig Configured strategy
     * @param decodeLoadBalanceStrategy DECODE role strategy configuration
     * @param vitLoadBalanceStrategy VIT role strategy configuration
     * @return Load balancing strategy to use for this role
     */
    public LoadBalanceStrategyEnum getStrategy(LoadBalanceStrategyEnum loadBalanceStrategyByConfig,
                                                LoadBalanceStrategyEnum decodeLoadBalanceStrategy,
                                                LoadBalanceStrategyEnum vitLoadBalanceStrategy) {
        // DECODE uses configured decodeLoadBalanceStrategy
        if (this == DECODE) {
            return decodeLoadBalanceStrategy != null ? decodeLoadBalanceStrategy : WEIGHTED_CACHE;
        }
        // VIT uses configured vitLoadBalanceStrategy
        if (this == VIT) {
            return vitLoadBalanceStrategy != null ? vitLoadBalanceStrategy : this.defaultStrategy;
        }
        // Other roles use configured strategy
        return loadBalanceStrategyByConfig != null ? loadBalanceStrategyByConfig : this.defaultStrategy;
    }

    /**
     * Get corresponding error type based on role type
     *
     * @return Corresponding error type
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
