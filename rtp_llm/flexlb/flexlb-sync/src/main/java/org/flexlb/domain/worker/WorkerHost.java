package org.flexlb.domain.worker;

import lombok.Getter;

/**
 * @author zjw
 * description:
 * date: 2025/4/24
 */
@Getter
public class WorkerHost {

    private final String ip;
    private final int httpPort;
    private final int grpcPort;         // usually grpcPort = httpPort + 1

    private final int httpServerPort;   // c++ http port
    private final String site;

    private final String group;

    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort, String site, String group) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.httpServerPort = httpServerPort;
        this.site = site;
        this.group = group;
    }

    public String getIpPort() {
        return ip + ":" + httpPort;
    }
}
