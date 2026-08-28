package org.flexlb.dao.loadbalance;

import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Result of a multi-role routing attempt.
 *
 * <p>A failed result may retain the workers selected before
 * {@code failedRoleType} could not be resolved.
 */
public record RoutingResult(boolean success,
                            List<ServerStatus> serverStatusList,
                            RoleType failedRoleType,
                            String errorMessage) {

    public static RoutingResult success(List<ServerStatus> serverStatusList) {
        return new RoutingResult(true, serverStatusList, null, null);
    }

    public static RoutingResult failure(List<ServerStatus> partialResults,
                                        RoleType failedRoleType,
                                        String errorMessage) {
        return new RoutingResult(
                false, partialResults, failedRoleType, errorMessage);
    }
}
