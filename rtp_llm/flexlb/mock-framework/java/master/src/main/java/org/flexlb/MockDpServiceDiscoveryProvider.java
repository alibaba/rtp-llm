package org.flexlb;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.util.JsonUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Expands a VIP member into independent Mock DP endpoints on the same Pod.
 */
final class MockDpServiceDiscoveryProvider implements ServiceDiscoveryProvider {
    private static final String DP_FIELD = "mock_dp_per_pod";

    private static final int PORT_STRIDE = 10;

    private static final Set<String> SUPPORTED_ROLES =
            Set.of("prefill_endpoint", "decode_endpoint");

    private final ServiceDiscoveryProvider delegate;
    private final Map<String, Integer> dpByVip;

    MockDpServiceDiscoveryProvider(
            ServiceDiscoveryProvider delegate, Map<String, Integer> dpByVip) {
        this.delegate = delegate;
        this.dpByVip = Map.copyOf(dpByVip);
    }

    @Override
    public ServiceDiscoveryType getType() {
        return delegate.getType();
    }

    @Override
    public void validate(Endpoint endpoint) {
        delegate.validate(endpoint);
    }

    @Override
    public List<WorkerHost> getHosts(Endpoint endpoint) {
        return expand(endpoint, delegate.getHosts(endpoint));
    }

    @Override
    public void listen(Endpoint endpoint, ServiceHostListener listener) {
        delegate.listen(
                endpoint,
                listener == null
                        ? null
                        : hosts -> listener.onHostsChanged(expand(endpoint, hosts)));
    }

    @Override
    public void shutdown() {
        delegate.shutdown();
    }

    static Map<String, Integer> readLayout(String document) throws Exception {
        if (document == null || document.isBlank()) {
            return Map.of();
        }
        Map<String, Integer> dpByVip = new HashMap<>();
        for (JsonNode group : JsonUtils.readStrictTree(document).path("role_endpoints")) {
            group.fields()
                    .forEachRemaining(
                            entry -> {
                                if (entry.getKey().endsWith("_endpoint")
                                        && entry.getValue().isObject()) {
                                    readEndpoint(entry.getKey(), entry.getValue(), dpByVip);
                                }
                            });
        }
        dpByVip.values().removeIf(dp -> dp == 1);
        return Map.copyOf(dpByVip);
    }

    private static void readEndpoint(String role, JsonNode endpoint, Map<String, Integer> dpByVip) {
        JsonNode value = endpoint.path(DP_FIELD);
        int dp = 1;
        if (!value.isMissingNode()) {
            if (!SUPPORTED_ROLES.contains(role)) {
                throw new IllegalArgumentException(DP_FIELD + " is not supported on " + role);
            }
            if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < 1) {
                throw new IllegalArgumentException(
                        DP_FIELD + " must be a positive integer on " + role);
            }
            dp = value.intValue();
        }

        String vip = endpoint.path("address").asText();
        boolean vipserver =
                "vipserver".equalsIgnoreCase(endpoint.path("discovery").path("type").asText());
        if (dp > 1
                && (!vipserver
                        || vip.isBlank()
                        || !"http".equalsIgnoreCase(endpoint.path("protocol").asText())
                        || endpoint.path("multi_engine_num").asInt(1) != 1
                        || endpoint.hasNonNull("worker_status_port"))) {
            throw new IllegalArgumentException(
                    DP_FIELD
                            + " requires an HTTP VIP with independent worker status ports: "
                            + vip);
        }
        if (vipserver && !vip.isBlank()) {
            Integer previous = dpByVip.putIfAbsent(vip, dp);
            if (previous != null && previous != dp) {
                throw new IllegalArgumentException(
                        "Conflicting " + DP_FIELD + " values for " + vip);
            }
        }
    }

    private List<WorkerHost> expand(Endpoint endpoint, List<WorkerHost> hosts) {
        int dp = dpByVip.getOrDefault(endpoint.getAddress(), 1);
        if (dp == 1 || hosts == null || hosts.isEmpty()) {
            return hosts;
        }

        List<WorkerHost> expanded = new ArrayList<>();
        for (WorkerHost host : hosts) {
            long lastPort = host.getPort() + (dp - 1L) * PORT_STRIDE;
            // The existing discovery router also derives the HTTP server port at base + 5.
            if (host.getPort() <= 0 || lastPort + 5 > 65535) {
                throw new IllegalArgumentException(
                        "Mock DP ports out of range for " + host.getIpPort());
            }
            expanded.add(host);
            for (int rank = 1; rank < dp; rank++) {
                int port = host.getPort() + rank * PORT_STRIDE;
                expanded.add(
                        WorkerHost.of(
                                host.getIp(), port, host.getSite(), host.getDeploymentName()));
            }
        }
        return expanded;
    }
}
