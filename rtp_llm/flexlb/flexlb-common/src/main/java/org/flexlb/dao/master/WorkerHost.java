package org.flexlb.dao.master;

import lombok.AccessLevel;
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
     * Per-engine gRPC port for worker control RPCs, including GetWorkerStatus and GetCacheStatus.
     */
    private final int workerStatusPort;
    /**
     * Logical engine index behind the shared frontend.
     */
    private final int engineIndex;
    /**
     * Expected number of logical engines for this physical frontend.
     */
    private final int multiEngineNum;
    /**
     * Canonical identity for this logical worker. It precomputes the physical frontend
     * identity ({@code ip:port}), routable/cache identity ({@code ip:port@engineIndex}), and
     * metrics identity ({@code ip@engineIndex}); callers use the corresponding semantic getters
     * exposed by {@link WorkerHost}.
     */
    @Getter(AccessLevel.NONE)
    private final WorkerIdentity workerIdentity;
    /**
     * Endpoint configuration address that produced this host.
     */
    private final String endpointAddress;
    /**
     * Data center/site information
     */
    private final String site;
    /**
     * Worker group name
     */
    private final String group;

    /**
     * Deployment name associated with this discovered instance, when provided by
     * the discovery backend.
     */
    private final String deploymentName;

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
        this(ip, httpPort, grpcPort, httpServerPort, grpcPort, site, group, "");
    }

    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort,
                      String site, String group, String deploymentName) {
        this(ip, httpPort, grpcPort, httpServerPort, grpcPort, site, group, deploymentName);
    }

    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort, int workerStatusPort,
                      String site, String group, String deploymentName) {
        this(ip, httpPort, grpcPort, httpServerPort, workerStatusPort,
                site, group, deploymentName, 0, 1, "");
    }

    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort, int workerStatusPort,
                      String site, String group, String deploymentName,
                      int engineIndex, int multiEngineNum) {
        this(ip, httpPort, grpcPort, httpServerPort, workerStatusPort,
                site, group, deploymentName, engineIndex, multiEngineNum, "");
    }

    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort, int workerStatusPort,
                      String site, String group, String deploymentName,
                      int engineIndex, int multiEngineNum, String endpointAddress) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.httpServerPort = httpServerPort;
        this.workerStatusPort = workerStatusPort;
        this.engineIndex = engineIndex;
        this.multiEngineNum = multiEngineNum;
        this.workerIdentity = new WorkerIdentity(ip, httpPort, engineIndex);
        this.endpointAddress = endpointAddress != null ? endpointAddress : "";
        this.site = site != null ? site : "";
        this.group = group != null ? group : "";
        this.deploymentName = deploymentName != null ? deploymentName : "";
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
     * Get the physical frontend address.
     *
     * @return physical address in {@code ip:port} format, without an engine index
     */
    public String getIpPort() {
        return workerIdentity.getPhysicalIpPort();
    }

    /** Returns the physical frontend address in {@code ip:port} format. */
    public String getPhysicalIpPort() {
        return workerIdentity.getPhysicalIpPort();
    }

    /**
     * Returns the logical worker identity in {@code ip:port@engineIndex} format. The index
     * identifies one independently routable engine behind the physical frontend.
     */
    public String getLogicalIpPort() {
        return workerIdentity.getLogicalIpPort();
    }

    /** Returns the port-free metrics identity in {@code ip@engineIndex} format. */
    public String getIpIndex() {
        return workerIdentity.getIpIndex();
    }

    public String getPhysicalGroupKey() {
        return endpointAddress + "|" + group + "|" + getPhysicalIpPort();
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

    public static WorkerHost of(String ip, int port, String site, String deploymentName) {
        return new WorkerHost(ip, port, port + 1, port + 5, site, "", deploymentName);
    }
}
