package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.HashMap;
import java.util.Map;

@Getter
public enum RoleType {
    PDFUSION("PDFUSION", "Prefill-Decode Fusion"),
    PREFILL("PREFILL", "Prefill"),
    DECODE("DECODE", "Decode"),
    VIT("VIT", "Vision Transformer");

    @JsonValue
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
     * Deserialize from JSON string. Accepts short name ("PREFILL") or proto-prefixed name ("ROLE_TYPE_PREFILL").
     */
    @JsonCreator
    public static RoleType fromString(String value) {
        if (value == null) {
            return null;
        }
        // Try code first ("PREFILL")
        RoleType byCode = CODE_MAP.get(value);
        if (byCode != null) {
            return byCode;
        }
        // Compat: strip proto prefix ("ROLE_TYPE_PREFILL" -> "PREFILL")
        if (value.startsWith("ROLE_TYPE_")) {
            return RoleType.valueOf(value.substring(10));
        }
        // Fallback: try enum name
        return RoleType.valueOf(value);
    }

    /**
     * Get corresponding error type based on role type.
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
