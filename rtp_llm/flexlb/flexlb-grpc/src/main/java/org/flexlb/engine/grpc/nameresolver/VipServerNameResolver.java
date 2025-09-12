package org.flexlb.engine.grpc.nameresolver;

import com.taobao.vipserver.client.core.Host;
import com.taobao.vipserver.client.core.HostListener;
import com.taobao.vipserver.client.core.VIPClient;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.util.JsonUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * @author zjw
 * description:
 * date: 2025/4/18
 */
@Slf4j
@Component
public class VipServerNameResolver implements CustomNameResolver {

    private final Map<String/*vipAddress*/, List<String/*ip:port*/>> domainHostsMap = new ConcurrentHashMap<>();
    private Listener listener;
    private List<String/*ip:port*/> allHosts = new ArrayList<>();

    public VipServerNameResolver(@Value("${MODEL_SERVICE_CONFIG:}") String modelConfig) {
        List<String> vipserverDomainList = initVipserverDomain(modelConfig);
        log.warn("VipServerNameResolver start subscribe clusters:{} ", vipserverDomainList);

        // 为每个域名创建独立的监听器
        for (String vipserverDomain : vipserverDomainList) {
            if (vipserverDomain == null) {
                log.warn("Skipping null vipserverDomain");
                continue;
            }
            HostListener domainListener = list -> updateDomainHosts(vipserverDomain, list);
            VIPClient.listen(vipserverDomain, domainListener);
        }
    }

    private List<String> initVipserverDomain(String modelConfigJson) {
        return Optional.ofNullable(modelConfigJson)
                .filter(StringUtils::isNotBlank)
                .map(json -> JsonUtils.toObject(modelConfigJson, ServiceRoute.class))
                .map(serviceRoute -> serviceRoute.getAllEndpoints().stream()
                        .filter(endpoint -> LoadBalanceStrategyEnum.VIPSERVER.getName().equals(endpoint.getType()))
                        .map(Endpoint::getAddress)
                        .collect(Collectors.toList()))
                .filter(CollectionUtils::isNotEmpty)
                .orElseThrow(() -> new IllegalArgumentException("vipserverDomainList cannot be null, please config 'MODEL_SERVICE_CONFIG' environment variable."));
    }

    @Override
    public void start(Listener listener) {
        this.listener = listener;
        listener.onAddressUpdate(allHosts);
    }

    /**
     * 更新指定域名的机器列表，并聚合所有域名的机器列表
     */
    private void updateDomainHosts(String domain, List<Host> hostList) {
        if (hostList == null || hostList.isEmpty()) {
            domainHostsMap.remove(domain);
        } else {
            List<String/*ip:port*/> domainHosts = new ArrayList<>(hostList.size());
            for (Host host : hostList) {
                domainHosts.add(host.getIp() + ":" + host.getPort());
            }
            domainHostsMap.put(domain, domainHosts);
        }

        // 聚合所有域名的机器列表
        List<String/*ip:port*/> aggregatedHosts = new ArrayList<>();
        for (List<String/*ip:port*/> hosts : domainHostsMap.values()) {
            aggregatedHosts.addAll(hosts);
        }

        log.info("Domain {} hosts updated, total aggregated hosts: {}", domain, aggregatedHosts.size());

        // 更新全局机器列表并通知监听器
        this.allHosts = aggregatedHosts;
        if (this.listener != null) {
            this.listener.onAddressUpdate(allHosts);
        }
    }


}
