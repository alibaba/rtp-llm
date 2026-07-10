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
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Service discovery backed by static host lists from environment variables.
 *
 * <p>For an address such as {@code com.example.prefill}, configure:
 * {@code FLEXLB_DISCOVERY_STATIC_HOSTS_COM_EXAMPLE_PREFILL=127.0.0.1:8080}.
 * The legacy {@code DOMAIN_ADDRESS:<address>} format remains supported.
 *
 * @author saichen.sm
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Slf4j
public final class StaticEnvironmentServiceDiscovery implements ServiceDiscovery {

    private static final StaticEnvironmentServiceDiscovery INSTANCE =
            new StaticEnvironmentServiceDiscovery();
    private static final String ENV_STATIC_HOSTS_PREFIX = "FLEXLB_DISCOVERY_STATIC_HOSTS_";
    private static final String LEGACY_ENV_DOMAIN_ADDRESS_PREFIX = "DOMAIN_ADDRESS:";

    public static StaticEnvironmentServiceDiscovery getInstance() {
        return INSTANCE;
    }

    @Override
    public List<WorkerHost> getHosts(String address) {
        if (StringUtils.isBlank(address)) {
            log.warn("Service address is blank, returning empty host list");
            return Collections.emptyList();
        }

        String envKey = environmentVariableName(address);
        String hostsConfig = System.getenv(envKey);
        if (StringUtils.isBlank(hostsConfig)) {
            String legacyEnvKey = LEGACY_ENV_DOMAIN_ADDRESS_PREFIX + address;
            hostsConfig = System.getenv(legacyEnvKey);
            if (StringUtils.isNotBlank(hostsConfig)) {
                log.warn("Environment variable {} is deprecated; use {} instead", legacyEnvKey, envKey);
            }
        }

        if (StringUtils.isBlank(hostsConfig)) {
            log.warn("No static hosts configured for address: {}, expected env var: {}", address, envKey);
            return Collections.emptyList();
        }

        try {
            return Arrays.stream(hostsConfig.split(","))
                    .map(String::trim)
                    .filter(StringUtils::isNotBlank)
                    .map(this::parseHost)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to parse static hosts for address: {}, config: {}", address, hostsConfig, e);
            return new ArrayList<>();
        }
    }

    @Override
    public void listen(String address, ServiceHostListener listener) {
        log.info("Static environment discovery does not support dynamic updates for address: {}", address);
        if (listener != null) {
            listener.onHostsChanged(getHosts(address));
        }
    }

    @Override
    public void shutdown() {
        log.info("StaticEnvironmentServiceDiscovery shutdown completed");
    }

    static String environmentVariableName(String address) {
        String normalizedAddress = address.trim()
                .toUpperCase(Locale.ROOT)
                .replaceAll("[^A-Z0-9]+", "_")
                .replaceAll("^_+|_+$", "");
        return ENV_STATIC_HOSTS_PREFIX + normalizedAddress;
    }

    private WorkerHost parseHost(String hostConfig) {
        int separator = hostConfig.lastIndexOf(':');
        if (separator <= 0 || separator == hostConfig.length() - 1) {
            throw new IllegalArgumentException("Invalid host format: " + hostConfig + ", expected host:port");
        }

        String host = hostConfig.substring(0, separator).trim();
        int port = Integer.parseInt(hostConfig.substring(separator + 1).trim());
        if (StringUtils.isBlank(host) || port < 1 || port > 65535) {
            throw new IllegalArgumentException("Invalid host:port value: " + hostConfig);
        }
        if (host.startsWith("[") && host.endsWith("]")) {
            host = host.substring(1, host.length() - 1);
        }
        return WorkerHost.of(host, port);
    }
}
