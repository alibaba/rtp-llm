package org.flexlb.engine.grpc;

import org.flexlb.dao.route.RoleType;

/**
 * Bidirectional converter between {@link RoleType} (domain enum) and
 * {@link EngineRpcService.RoleTypePB} (proto-generated enum).
 *
 * <p>Lives in flexlb-grpc (not flexlb-common) to avoid a reverse dependency:
 * flexlb-common must not depend on flexlb-grpc.</p>
 */
public final class RoleTypeProtoConverter {

    private RoleTypeProtoConverter() {
    }

    /**
     * Convert proto enum to domain {@link RoleType}.
     */
    public static RoleType fromProto(EngineRpcService.RoleTypePB proto) {
        return switch (proto) {
            case ROLE_TYPE_PDFUSION -> RoleType.PDFUSION;
            case ROLE_TYPE_PREFILL -> RoleType.PREFILL;
            case ROLE_TYPE_DECODE -> RoleType.DECODE;
            case ROLE_TYPE_VIT -> RoleType.VIT;
            case ROLE_TYPE_FRONTEND -> RoleType.FRONTEND;
            default -> null;
        };
    }

    /**
     * Convert domain {@link RoleType} to proto enum.
     */
    public static EngineRpcService.RoleTypePB toProto(RoleType role) {
        return switch (role) {
            case PDFUSION -> EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;
            case PREFILL -> EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL;
            case DECODE -> EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE;
            case VIT -> EngineRpcService.RoleTypePB.ROLE_TYPE_VIT;
            case FRONTEND -> EngineRpcService.RoleTypePB.ROLE_TYPE_FRONTEND;
        };
    }
}
