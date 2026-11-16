package org.flexlb.worker.status;

import lombok.Getter;

/**
 * Addresses one logical Worker engine and its shared frontend.
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
     * Per-engine gRPC port for worker control RPCs.
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

    private final String physicalIpPort;
    private final String logicalIpPort;

    /**
     * Data center/site information
     */
    private final String site;

    /**
     * Worker group name
     */
    private final String group;

    /**
     * Deployment name reported by service discovery.
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
    public WorkerHost(
            String ip, int httpPort, int grpcPort, int httpServerPort, String site, String group) {
        this(ip, httpPort, grpcPort, httpServerPort, grpcPort, site, group, "");
    }

    public WorkerHost(
            String ip,
            int httpPort,
            int grpcPort,
            int httpServerPort,
            String site,
            String group,
            String deploymentName) {
        this(ip, httpPort, grpcPort, httpServerPort, grpcPort, site, group, deploymentName);
    }

    public WorkerHost(
            String ip,
            int httpPort,
            int grpcPort,
            int httpServerPort,
            int workerStatusPort,
            String site,
            String group,
            String deploymentName) {
        this(
                ip,
                httpPort,
                grpcPort,
                httpServerPort,
                workerStatusPort,
                site,
                group,
                deploymentName,
                0,
                1);
    }

    public WorkerHost(
            String ip,
            int httpPort,
            int grpcPort,
            int httpServerPort,
            int workerStatusPort,
            String site,
            String group,
            String deploymentName,
            int engineIndex,
            int multiEngineNum) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.httpServerPort = httpServerPort;
        this.workerStatusPort = workerStatusPort;
        this.engineIndex = engineIndex;
        this.multiEngineNum = multiEngineNum;
        this.physicalIpPort = ip == null ? null : ip + ":" + httpPort;
        this.logicalIpPort = physicalIpPort == null ? null : physicalIpPort + "@" + engineIndex;
        this.site = site != null ? site : "";
        this.group = group != null ? group : "";
        this.deploymentName = deploymentName != null ? deploymentName : "";
    }

    public String getIpPort() {
        return physicalIpPort;
    }

    public int getPort() {
        return httpPort;
    }
}
