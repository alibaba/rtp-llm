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
    VIT("VIT", "Vision Transformer"),
    FRONTEND("FRONTEND", "Frontend");

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
            try {
                return RoleType.valueOf(value.substring(10));
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
        // Fallback: try enum name
        try {
            return RoleType.valueOf(value);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    /**
     * Check if string matches current role type.
     *
     * @deprecated Use {@code roleType == RoleType.PREFILL} or enum comparison instead.
     */
    @Deprecated
    public boolean matches(String code) {
        return this.code.equals(code);
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
            case FRONTEND -> StrategyErrorType.NO_FRONTEND_WORKER;
        };
    }

    /**
     * Get the proto enum constant name (ROLE_TYPE_XXX).
     *
     * @deprecated Use {@link org.flexlb.engine.grpc.RoleTypeProtoConverter#toProto(RoleType)}
     *             for direct proto enum mapping.
     */
    @Deprecated
    public String getProtoName() {
        return "ROLE_TYPE_" + this.name();
    }
}
