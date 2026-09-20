package org.flexlb.dao.loadbalance;

import java.util.List;

/**
 * Routing result
 *
 * @author saichen.sm
 * @since 2025/12/25
 */
public record RoutingResult(boolean success, List<ServerStatus> serverStatusList, ServerStatus failure) {

    public static RoutingResult success(List<ServerStatus> serverStatusList) {
        return new RoutingResult(true, serverStatusList, null);
    }

    public static RoutingResult failure(List<ServerStatus> partialResults, ServerStatus failure) {
        return new RoutingResult(false, partialResults, failure);
    }
}
