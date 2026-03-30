package org.flexlb.discovery;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
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

    @Override
    public List<WorkerHost> getHosts(String address) {
        if (StringUtils.isBlank(address)) {
            log.warn("Service address is blank, returning empty host list");
            return Collections.emptyList();
        }
        // Convert address to environment variable key (replace special characters)
        String envKey = ENV_DOMAIN_ADDRESS + address;
        String hostsConfig = System.getenv(envKey);
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
            log.error("Failed to parse hosts configuration for address: {}, config: {}", address, hostsConfig, e);
            return new ArrayList<>();
        }
    }

    @Override
    public void listen(String address, ServiceHostListener listener) {
        log.info("NoOpServiceDiscovery does not support dynamic listening for address: {}", address);
        // Default empty implementation does not support dynamic listening, could consider periodic polling implementation
        // Simply trigger initialization once here
        if (listener != null) {
            listener.onHostsChanged(getHosts(address));
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
