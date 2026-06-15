package org.flexlb.service.optimizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.listener.AppShutDownHooker;
import org.flexlb.transport.GeneralHttpNettyService;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Component
public class OnlineOptimizerHooker implements AppOnlineHooker, AppShutDownHooker {

    private final GeneralHttpNettyService httpService;
    private final ServiceDiscovery serviceDiscovery;

    private final String instanceGroup;
    private final String basePath;
    private final int registerTimeoutMs;
    private final String instanceId;
    private final String vipserverDomain;
    private final String directAddress;

    @Getter
    private final boolean enabled;

    @Getter
    private volatile OnlineOptimizerClient client;

    // Guard against duplicate afterStartUp invocations (e.g. sidecar retry on /hook/after_start)
    // to prevent ServiceDiscovery listener leaks and duplicate retry schedulers.
    private final AtomicBoolean started = new AtomicBoolean(false);

    public OnlineOptimizerHooker(GeneralHttpNettyService httpService, ServiceDiscovery serviceDiscovery) {

        this.httpService = httpService;
        this.serviceDiscovery = serviceDiscovery;
        this.instanceGroup = envOrDefault("ONLINE_OPTIMIZER_INSTANCE_GROUP", "");
        this.basePath = envOrDefault("ONLINE_OPTIMIZER_BASE_PATH", "/api/optimizer");
        this.registerTimeoutMs = intEnvOrDefault("ONLINE_OPTIMIZER_REGISTER_TIMEOUT_MS", 5000);
        this.vipserverDomain = envOrDefault("OPTIMIZER_VIPSERVER_DOMAIN", "");
        this.directAddress = envOrDefault("OPTIMIZER_DIRECT_ADDRESS", "");
        this.instanceId = !instanceGroup.isEmpty() ? resolveInstanceId() : "";

        boolean addressConfigured = !vipserverDomain.isEmpty() || !directAddress.isEmpty();
        boolean serviceDiscoveryReady = vipserverDomain.isEmpty() || serviceDiscovery != null;

        if (instanceGroup.isEmpty()) {
            log.info("OnlineOptimizer disabled");
            this.enabled = false;
        } else if (instanceId.isEmpty()) {
            log.warn("OnlineOptimizer disabled: instanceId is empty, please set ONLINE_OPTIMIZER_INSTANCE_ID " +
                    "or ensure MODEL_SERVICE_CONFIG contains a valid serviceId");
            this.enabled = false;
        } else if (!addressConfigured) {
            log.warn("OnlineOptimizer disabled: address resolver could not be created, " +
                    "check OPTIMIZER_VIPSERVER_DOMAIN / OPTIMIZER_DIRECT_ADDRESS / ServiceDiscovery bean");
            this.enabled = false;
        } else if (!serviceDiscoveryReady) {
            log.warn("OnlineOptimizer disabled: ServiceDiscovery bean required for vipserver domain={}", vipserverDomain);
            this.enabled = false;
        } else {
            this.enabled = true;
        }
    }

    @Override
    public void afterStartUp() {
        if (!enabled) return;
        if (!started.compareAndSet(false, true)) {
            log.info("OnlineOptimizer afterStartUp already executed, skip duplicate invocation");
            return;
        }

        // afterStartUp runs after Spring has fully started (GracefulOnlineService.online),
        // so blocking IO from getHosts/listen does not affect container startup.
        OptimizerAddressResolver resolver = createAddressResolver();
        if (resolver == null) {
            log.warn("OnlineOptimizer afterStartUp: address resolver could not be created, skip registration");
            return;
        }
        OnlineOptimizerClient newClient =
                new OnlineOptimizerClient(httpService, resolver, instanceGroup, basePath, registerTimeoutMs);
        this.client = newClient;

        int blockSize = intEnvOrDefault("ONLINE_OPTIMIZER_BLOCK_SIZE", 0);
        int blockBytes = intEnvOrDefault("ONLINE_OPTIMIZER_BLOCK_BYTES", 0);
        int linearStep = intEnvOrDefault("ONLINE_OPTIMIZER_LINEAR_STEP", 0);
        String fullGroupName = envOrDefault("ONLINE_OPTIMIZER_FULL_GROUP_NAME", "");

        List<OptimizerRegisterRequest.LocationSpecInfo> locationSpecInfos = parseLocationSpecInfos();
        if (locationSpecInfos == null && blockBytes > 0) {
            locationSpecInfos = List.of(new OptimizerRegisterRequest.LocationSpecInfo("full", blockBytes));
        }

        List<OptimizerRegisterRequest.LocationSpecGroup> locationSpecGroups = parseLocationSpecGroups();

        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .instanceGroup(instanceGroup)
                .blockSize(blockSize)
                .locationSpecInfos(locationSpecInfos)
                .locationSpecGroups(locationSpecGroups)
                .linearStep(linearStep)
                .fullGroupName(fullGroupName)
                .build();

        newClient.startRegistrationAsync(instanceId, params);
        log.info("OnlineOptimizer registration submitted (async): instanceId={}", instanceId);
    }

    @Override
    public void beforeShutdown() {
        if (client != null) {
            client.shutdown();
            log.info("OnlineOptimizer shutdown completed");
        }
    }

    @Override
    public int priority() {
        return 0;
    }

    private String resolveInstanceId() {
        String explicit = System.getenv("ONLINE_OPTIMIZER_INSTANCE_ID");
        if (explicit != null && !explicit.isEmpty()) return explicit;

        String modelConfig = System.getenv("MODEL_SERVICE_CONFIG");
        if (modelConfig != null && !modelConfig.isEmpty()) {
            try {
                ServiceRoute route = org.flexlb.util.JsonUtils.toObject(modelConfig, ServiceRoute.class);
                if (route != null && route.getServiceId() != null && !route.getServiceId().isEmpty()) {
                    return route.getServiceId() + "-master";
                }
            } catch (Exception e) {
                log.warn("Failed to parse MODEL_SERVICE_CONFIG for instanceId: {}", e.getMessage());
            }
        }

        return "";
    }

    private OptimizerAddressResolver createAddressResolver() {
        if (!vipserverDomain.isEmpty()) {
            // Defer start() to client's async retry; never block afterStartUp on listen failure.
            log.info("OnlineOptimizer using ServiceDiscoveryAddressResolver, domain={}", vipserverDomain);
            return new ServiceDiscoveryAddressResolver(serviceDiscovery, vipserverDomain);
        }
        if (!directAddress.isEmpty()) {
            return new DirectAddressResolver(directAddress);
        }
        return null;
    }

    private static String envOrDefault(String key, String defaultValue) {
        String val = System.getenv(key);
        return (val != null && !val.isEmpty()) ? val : defaultValue;
    }

    private static int intEnvOrDefault(String key, int defaultValue) {
        String val = System.getenv(key);
        if (val != null && !val.isEmpty()) {
            try {
                return Integer.parseInt(val);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    /**
     * Parse ONLINE_OPTIMIZER_LOCATION_SPEC_INFOS, format: "name1:size1,name2:size2"
     * e.g. "full:131072,linear:65536"
     */
    private static List<OptimizerRegisterRequest.LocationSpecInfo> parseLocationSpecInfos() {
        String val = System.getenv("ONLINE_OPTIMIZER_LOCATION_SPEC_INFOS");
        if (val == null || val.isEmpty()) return null;

        List<OptimizerRegisterRequest.LocationSpecInfo> result = new ArrayList<>();
        for (String part : val.split(",")) {
            String[] kv = part.trim().split(":");
            if (kv.length == 2) {
                try {
                    result.add(new OptimizerRegisterRequest.LocationSpecInfo(kv[0].trim(), Integer.parseInt(kv[1].trim())));
                } catch (NumberFormatException e) {
                    log.warn("Invalid location_spec_info entry: {}", part);
                }
            }
        }
        return result.isEmpty() ? null : result;
    }

    /**
     * Parse ONLINE_OPTIMIZER_LOCATION_SPEC_GROUPS, format: "groupName:spec1|spec2,groupName2:spec3|spec4"
     * e.g. "full_group:full|linear"
     */
    private static List<OptimizerRegisterRequest.LocationSpecGroup> parseLocationSpecGroups() {
        String val = System.getenv("ONLINE_OPTIMIZER_LOCATION_SPEC_GROUPS");
        if (val == null || val.isEmpty()) return null;

        List<OptimizerRegisterRequest.LocationSpecGroup> result = new ArrayList<>();
        for (String part : val.split(",")) {
            String[] kv = part.trim().split(":");
            if (kv.length == 2) {
                String groupName = kv[0].trim();
                List<String> specNames = Arrays.asList(kv[1].trim().split("\\|"));
                result.add(new OptimizerRegisterRequest.LocationSpecGroup(groupName, specNames));
            }
        }
        return result.isEmpty() ? null : result;
    }
}
