package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;

public abstract class WorkerEndpoint {

    private final String ip;
    private final int httpPort;
    private final int grpcPort;
    private final WorkerStatus status;

    protected WorkerEndpoint(String ip, int httpPort, int grpcPort, WorkerStatus status) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.status = status;
    }

    public String ipPort() {
        return ip + ":" + httpPort;
    }

    public String getIp() {
        return ip;
    }

    public int getHttpPort() {
        return httpPort;
    }

    public int getGrpcPort() {
        return grpcPort;
    }

    public WorkerStatus getStatus() {
        return status;
    }
}
