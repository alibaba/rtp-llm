package org.flexlb.dao.master;

import lombok.Getter;

/**
 * WorkerHost - Worker node host information
 * Unified host information representation for service discovery and worker management
 *
 * @author saichen.sm
 * @since 2025/4/24
 */
@Getter
public class WorkerHost {

    /**
     * Host IP address
     */
    private final String ip;
    /**
     * HTTP port
     */
    private final int httpPort;
    /**
     * gRPC port (typically httpPort + 1)
     */
    private final int grpcPort;
    /**
     * C++ HTTP service port
     */
    private final int httpServerPort;
    /**
     * Data center/site information
     */
    private final String site;
    /**
     * Worker group name
     */
    private final String group;

    /**
     * Full constructor
     *
     * @param ip             Host IP address
     * @param httpPort       HTTP port
     * @param grpcPort       gRPC port
     * @param httpServerPort C++ HTTP service port
     * @param site           Data center/site information
     * @param group          Worker group name
     */
    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort, String site, String group) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.httpServerPort = httpServerPort;
        this.site = site != null ? site : "";
        this.group = group != null ? group : "";
    }

    /**
     * Simplified constructor (for service discovery scenarios)
     *
     * @param ip   Host IP address
     * @param port Main port
     * @param site Data center/site information
     */
    public WorkerHost(String ip, int port, String site) {
        this(ip, port, port + 1, port + 5, site, "");
    }

    /**
     * Minimal constructor (for basic service discovery scenarios)
     *
     * @param ip   Host IP address
     * @param port Main port
     */
    public WorkerHost(String ip, int port) {
        this(ip, port, "");
    }

    /**
     * Get IP:Port format string
     *
     * @return IP:Port format string
     */
    public String getIpPort() {
        return ip + ":" + httpPort;
    }

    /**
     * Get main port (typically HTTP port)
     *
     * @return Main port number
     */
    public int getPort() {
        return httpPort;
    }

    /**
     * Create WorkerHost instance
     *
     * @param ip   Host IP address
     * @param port Host port
     * @return WorkerHost instance
     */
    public static WorkerHost of(String ip, int port) {
        return new WorkerHost(ip, port);
    }

    /**
     * Create WorkerHost instance
     *
     * @param ip   Host IP address
     * @param port Host port
     * @param site Site information
     * @return WorkerHost instance
     */
    public static WorkerHost of(String ip, int port, String site) {
        return new WorkerHost(ip, port, site);
    }
}
