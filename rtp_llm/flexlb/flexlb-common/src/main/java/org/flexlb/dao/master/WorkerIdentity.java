package org.flexlb.dao.master;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;

import static org.flexlb.constant.CommonConstants.LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR;

/**
 * Immutable worker identity with all commonly used address representations precomputed.
 *
 * <ul>
 *   <li>{@code ip}: raw host IP</li>
 *   <li>{@code port}: raw shared frontend port</li>
 *   <li>{@code engineIndex}: raw logical engine index</li>
 *   <li>{@code physicalIpPort}: {@code ip:port}, identifying the shared frontend</li>
 *   <li>{@code logicalIpPort}: {@code ip:port@engineIndex}, used by routing and cache matching</li>
 *   <li>{@code ipIndex}: {@code ip@engineIndex}, used by per-engine metrics</li>
 * </ul>
 */
@Getter
@EqualsAndHashCode
@ToString
public final class WorkerIdentity {

    /** Raw host IP, without a port or engine index. */
    private final String ip;
    /** Raw shared frontend port. */
    private final int port;
    /** Raw logical engine index behind the shared frontend. */
    private final int engineIndex;
    /** Shared physical frontend identity in {@code ip:port} format. */
    private final String physicalIpPort;
    /** Routable/cache identity in {@code ip:port@engineIndex} format. */
    private final String logicalIpPort;
    /** Per-engine metrics identity in {@code ip@engineIndex} format. */
    private final String ipIndex;

    public WorkerIdentity(String ip, int port, int engineIndex) {
        this.ip = ip;
        this.port = port;
        this.engineIndex = engineIndex;
        this.physicalIpPort = ip == null ? null : ip + ":" + port;
        this.logicalIpPort = physicalIpPort == null
                ? null
                : physicalIpPort + LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR + engineIndex;
        this.ipIndex = ip == null
                ? null
                : ip + LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR + engineIndex;
    }
}
