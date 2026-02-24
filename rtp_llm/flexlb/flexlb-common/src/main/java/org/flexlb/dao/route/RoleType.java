package org.flexlb.dao.route;

import lombok.Getter;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.HashMap;
import java.util.Map;

@Getter
public enum RoleType {
    PDFUSION("RoleType.PDFUSION", "Prefill-Decode Fusion"),
    PREFILL("RoleType.PREFILL", "Prefill"),
    DECODE("RoleType.DECODE", "Decode"),
    VIT("RoleType.VIT", "Vision Transformer");

    private final String code;
    private final String description;

    private static final Map<String, RoleType> CODE_MAP = new HashMap<>();

    static {
        for (RoleType roleType : RoleType.values()) {
            CODE_MAP.put(roleType.code, roleType);
        }
    }

    RoleType(String code, String description) {
        this.code = code;
        this.description = description;
    }

    /**
     * Check if string matches current role type
     */
    public boolean matches(String code) {
        return this.code.equals(code);
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
