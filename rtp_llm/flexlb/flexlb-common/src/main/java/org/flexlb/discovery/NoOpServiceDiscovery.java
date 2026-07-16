package org.flexlb.discovery;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * NoOpServiceDiscovery - Default service discovery implementation
 *
 * @author saichen.sm
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Slf4j
public final class NoOpServiceDiscovery implements ServiceDiscovery {

    private static final NoOpServiceDiscovery INSTANCE = new NoOpServiceDiscovery();
    /**
     * Environment variable prefix for configuring host list corresponding to service address
     * Format: ip1:port1,ip2:port2
     */
    private static final String ENV_DOMAIN_ADDRESS = "DOMAIN_ADDRESS:";

    public static NoOpServiceDiscovery getInstance() {
        return INSTANCE;
    }

    /**
     * Resolve hosts for the given address from the {@code DOMAIN_ADDRESS:<address>} environment variable.
     *
     * <p>Contract: a missing (or blank) env value means an empty fleet, which is legal — logs a WARN
     * and returns an empty list. A malformed env value is a lookup failure, not an empty fleet, and
     * throws {@link IllegalArgumentException} so it cannot be swallowed as "no hosts".
     */
    @Override
    public List<WorkerHost> getHosts(String address) {
        return getHosts(address, System.getenv());
    }

    /**
     * Resolve against an explicit environment rather than the process environment.
     *
     * @param address Service address to resolve
     * @param env     Environment to read {@code DOMAIN_ADDRESS:<address>} from
     */
    List<WorkerHost> getHosts(String address, Map<String, String> env) {
        if (StringUtils.isBlank(address)) {
            log.warn("Service address is blank, returning empty host list");
            return Collections.emptyList();
        }
        // Convert address to environment variable key (replace special characters)
        String envKey = ENV_DOMAIN_ADDRESS + address;
        String hostsConfig = env.get(envKey);
        if (StringUtils.isBlank(hostsConfig)) {
            log.warn("No hosts configuration found for address: {}, expected env var: {}", address, envKey);
            return Collections.emptyList();
        }
        try {
            return Arrays.stream(hostsConfig.split(","))
                    .map(String::trim)
                    .filter(StringUtils::isNotBlank)
                    .map(this::parseHost)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            // Propagate instead of returning an empty list: callers treat "empty" as a genuinely
            // empty fleet (embedding liveness mass-kills on it), while a malformed DOMAIN_ADDRESS
            // is a lookup failure that must surface as ServiceDiscoveryException upstream.
            throw new IllegalArgumentException(
                    "malformed hosts configuration for address " + address + ": " + hostsConfig, e);
        }
    }

    /**
     * Registration, not a lookup: a caller wiring up a listener is not asking for hosts, so a
     * failed initial resolve is logged rather than thrown. The "malformed config throws" contract
     * belongs to {@link #getHosts(String)} and stays there.
     */
    @Override
    public void listen(String address, ServiceHostListener listener) {
        log.info("NoOpServiceDiscovery does not support dynamic listening for address: {}", address);
        // Default empty implementation does not support dynamic listening, could consider periodic polling implementation
        // Simply trigger initialization once here
        if (listener == null) {
            return;
        }
        try {
            listener.onHostsChanged(getHosts(address));
        } catch (Exception e) {
            log.error("NoOpServiceDiscovery initial host push failed for address: {}", address, e);
        }
    }

    @Override
    public void shutdown() {
        log.info("NoOpServiceDiscovery shutdown");
        // No operation
    }

    /**
     * Parse host string to WorkerHost object
     */
    private WorkerHost parseHost(String hostStr) {
        String[] parts = hostStr.split(":");
        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid host format: " + hostStr + ", expected ip:port");
        }
        String ip = parts[0].trim();
        int port = Integer.parseInt(parts[1].trim());
        return WorkerHost.of(ip, port);
    }
}
