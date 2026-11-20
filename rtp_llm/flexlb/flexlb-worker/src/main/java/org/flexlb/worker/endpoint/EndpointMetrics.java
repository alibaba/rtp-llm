package org.flexlb.worker.endpoint;

/**
 * Reports endpoint cleanup counts without depending on scheduler metrics.
 */
public interface EndpointMetrics {
    void reportEndpointInflightTtlExpired(String role, String engineIp, String reason, int count);
}
