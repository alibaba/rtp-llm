package org.flexlb.discovery;

import com.google.common.net.InetAddresses;
import org.flexlb.dao.master.WorkerHost;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

/**
 * Resolves raw service addresses for clients that connect to a bootstrap endpoint.
 */
public final class ServiceAddresses {
    private ServiceAddresses() {}

    /** Resolves an explicit host:port or a discovery name using the provider's ports. */
    public static List<String> resolve(String address, ServiceDiscovery discovery) {
        if (discovery == null) {
            return List.of(fixedAddress(address));
        }
        List<WorkerHost> hosts = discovery.getHosts(address);
        List<String> addresses = new ArrayList<>(hosts.size());
        for (WorkerHost host : hosts) {
            try {
                addresses.add(authority(host.getIp(), host.getPort()));
            } catch (IllegalStateException ignored) {
                // A bad discovery record must not hide the other candidates.
            }
        }
        return addresses;
    }

    /** Formats a raw network authority, including brackets for IPv6 literals. */
    public static String authority(String host, int port) {
        if (host == null || host.isBlank() || port < 1 || port > 65535) {
            throw new IllegalStateException("Invalid service endpoint");
        }
        try {
            return new URI(null, null, host, port, null, null, null).getRawAuthority();
        } catch (URISyntaxException error) {
            throw new IllegalStateException("Invalid service host", error);
        }
    }

    /** Uses the configured port with the first discovered host or the configured host. */
    public static String resolveFirst(String address, int port, ServiceDiscovery discovery) {
        if (InetAddresses.isInetAddress(address) || InetAddresses.isUriInetAddress(address)) {
            return authority(address, port);
        }
        List<WorkerHost> hosts = discovery.getHosts(address);
        String host = hosts.isEmpty() ? address : hosts.getFirst().getIp();
        return authority(host, port);
    }

    private static String fixedAddress(String address) {
        URI uri = URI.create("tcp://" + address);
        if (!address.equals(uri.getRawAuthority()) || uri.getRawUserInfo() != null
                || uri.getHost() == null || uri.getPort() < 1 || uri.getPort() > 65535) {
            throw new IllegalArgumentException("Expected a service host:port");
        }
        return uri.getRawAuthority();
    }
}
