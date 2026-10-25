package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.schedulev2.request.Request;

/**
 * 发布前确认 Prefill 仍是注册表中的同一台节点，且尚未开始下线。hold 保留到本次发布完成。
 */
public final class PrefillPublishCheck {

    private final EndpointRegistry endpoints;

    public PrefillPublishCheck(EndpointRegistry endpoints) {
        this.endpoints = endpoints;
    }

    public boolean isAvailable(Request request) {
        PrefillEndpoint endpoint = request.prefillChoice().endpoint();
        return !endpoint.isRetired() && endpoints.get(endpoint.getStatus().getRole(), endpoint.ipPort()) == endpoint;
    }

    /**
     * 节点准备下线、已移除或同地址已被替换时返回 null。
     */
    public WorkerEndpoint.EndpointHold hold(Request request) {
        PrefillEndpoint endpoint = request.prefillChoice().endpoint();
        WorkerEndpoint.EndpointHold hold = endpoints.hold(endpoint.getStatus().getRole(), endpoint.ipPort());
        if (hold == null) {
            return null;
        }
        if (hold.endpoint() != endpoint || endpoint.isRetired()) {
            hold.close();
            return null;
        }
        return hold;
    }
}
