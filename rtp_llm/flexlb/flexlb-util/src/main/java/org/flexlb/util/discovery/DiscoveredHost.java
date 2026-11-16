package org.flexlb.util.discovery;

/**
 * An address returned by a discovery provider, without derived service ports or engine identities.
 */
@lombok.Getter
public final class DiscoveredHost {
    private final String ip;
    private final int port;
    private final String site;
    private final String deploymentName;

    public DiscoveredHost(String ip, int port) {
        this(ip, port, "");
    }

    public DiscoveredHost(String ip, int port, String site) {
        this(ip, port, site, "");
    }

    public DiscoveredHost(String ip, int port, String site, String deploymentName) {
        this.ip = ip;
        this.port = port;
        this.site = site == null ? "" : site;
        this.deploymentName = deploymentName == null ? "" : deploymentName;
    }

    public String getIpPort() {
        return ip + ":" + port;
    }

    public static DiscoveredHost of(String ip, int port) {
        return new DiscoveredHost(ip, port);
    }

    public static DiscoveredHost of(String ip, int port, String site) {
        return new DiscoveredHost(ip, port, site);
    }

    public static DiscoveredHost of(String ip, int port, String site, String deploymentName) {
        return new DiscoveredHost(ip, port, site, deploymentName);
    }
}
