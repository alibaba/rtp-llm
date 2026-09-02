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

    /** Convert a domain role to the current protocol enum. */
    public static EngineRpcService.RoleTypePB toProto(RoleType role) {
        return switch (role) {
            case PDFUSION -> EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;
            case PREFILL -> EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL;
            case DECODE -> EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE;
            case VIT -> EngineRpcService.RoleTypePB.ROLE_TYPE_VIT;
            case FRONTEND -> EngineRpcService.RoleTypePB.ROLE_TYPE_FRONTEND;
        };
    }

    /** Convert the domain role to the original RoleAddrPB field-1 enum. */
    public static EngineRpcService.RoleAddrPB.RoleType toLegacyProto(RoleType role) {
        return switch (role) {
            case PDFUSION -> EngineRpcService.RoleAddrPB.RoleType.PDFUSION;
            case PREFILL -> EngineRpcService.RoleAddrPB.RoleType.PREFILL;
            case DECODE -> EngineRpcService.RoleAddrPB.RoleType.DECODE;
            case VIT -> EngineRpcService.RoleAddrPB.RoleType.VIT;
            case FRONTEND -> EngineRpcService.RoleAddrPB.RoleType.FRONTEND;
        };
    }

    /** Original opaque WorkerStatusPB field-1 spelling used by old Masters. */
    public static String toLegacyWorkerStatusString(RoleType role) {
        return "RoleType." + role.name();
    }

    /** Read RoleAddrPB across dsv4 and dual-write schemas. */
    public static RoleType fromRoleAddr(EngineRpcService.RoleAddrPB addr) {
        RoleType resolved = null;
        if (!addr.getRoleStr().isEmpty()) {
            resolved = requireKnown(RoleType.fromString(addr.getRoleStr()),
                    "role_str='" + addr.getRoleStr() + "'");
        }
        if (addr.getRole() != EngineRpcService.RoleAddrPB.RoleType.PDFUSION) {
            resolved = merge(resolved, fromLegacyProto(addr.getRole()), "role=" + addr.getRole());
        }
        // The original proto3 enum omitted PDFUSION=0 from the wire.
        return resolved != null ? resolved : RoleType.PDFUSION;
    }

    /** Read WorkerStatusPB across the legacy string and typed enum schemas. */
    public static RoleType fromWorkerStatus(EngineRpcService.WorkerStatusPB status) {
        RoleType resolved = null;
        if (status.getRoleType() != EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION) {
            resolved = requireKnown(fromProto(status.getRoleType()),
                    "role_type=" + status.getRoleType());
        }
        String name = status.getRole();
        if (name != null && !name.isEmpty()) {
            resolved = merge(resolved,
                    requireKnown(RoleType.fromString(name), "role='" + name + "'"),
                    "role='" + name + "'");
        }
        // The original proto3 enum omitted PDFUSION=0 from the wire.
        return resolved != null ? resolved : RoleType.PDFUSION;
    }

    private static RoleType requireKnown(RoleType role, String source) {
        if (role == null) {
            throw new IllegalArgumentException("unknown RoleAddrPB " + source);
        }
        return role;
    }

    private static RoleType merge(RoleType resolved, RoleType candidate, String source) {
        if (resolved != null && resolved != candidate) {
            throw new IllegalArgumentException("conflicting RoleAddrPB role from " + source
                    + ": resolved=" + resolved + ", candidate=" + candidate);
        }
        return candidate;
    }

    private static RoleType fromLegacyProto(EngineRpcService.RoleAddrPB.RoleType role) {
        return switch (role) {
            case PDFUSION -> RoleType.PDFUSION;
            case PREFILL -> RoleType.PREFILL;
            case DECODE -> RoleType.DECODE;
            case VIT -> RoleType.VIT;
            case FRONTEND -> RoleType.FRONTEND;
            case UNRECOGNIZED -> throw new IllegalArgumentException("unknown legacy RoleAddrPB role: " + role);
        };
    }
}
