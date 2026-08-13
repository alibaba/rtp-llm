package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.OnlineOptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.util.JsonUtils;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Loads and validates the model routing configuration once during startup.
 */
@Slf4j
@Configuration
public class ModelServiceConfiguration {

    private static final String MODEL_SERVICE_CONFIG = "MODEL_SERVICE_CONFIG";

    @Bean
    public ModelMetaConfig modelMetaConfig(Environment environment, RoutingServiceDiscovery serviceDiscovery) {
        String modelConfigJson = environment.getProperty(MODEL_SERVICE_CONFIG);
        if (StringUtils.isBlank(modelConfigJson)) {
            throw new IllegalStateException(MODEL_SERVICE_CONFIG + " must not be blank");
        }

        ServiceRoute serviceRoute = JsonUtils.toObject(modelConfigJson, ServiceRoute.class);
        validateServiceRoute(serviceRoute, serviceDiscovery);

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        log.info("Loaded model service route: serviceId={}, endpoints={}",
                serviceRoute.getServiceId(), serviceRoute.getAllEndpoints().size());
        return modelMetaConfig;
    }

    private void validateServiceRoute(ServiceRoute serviceRoute, RoutingServiceDiscovery serviceDiscovery) {
        if (StringUtils.isBlank(serviceRoute.getServiceId())) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG service_id must not be blank");
        }
        if (CollectionUtils.isEmpty(serviceRoute.getRoleEndpoints())) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG role_endpoints must not be empty");
        }

        var endpoints = serviceRoute.getAllEndpoints();
        if (CollectionUtils.isEmpty(endpoints)) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG must contain at least one role endpoint");
        }
        for (Endpoint endpoint : endpoints) {
            serviceDiscovery.validate(endpoint);
        }

        validateKvcm(serviceRoute.getKvcm(), serviceDiscovery);
        validateOnlineOptimizer(serviceRoute.getOnlineOptimizer(), serviceDiscovery);
    }

    private void validateKvcm(KvcmConfig kvcm, RoutingServiceDiscovery serviceDiscovery) {
        if (kvcm == null || !kvcm.isEnabled()) {
            return;
        }
        if (kvcm.getRequestTimeoutMs() <= 0 || kvcm.getLeaderRefreshIntervalMs() <= 0) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG kvcm timeouts must be greater than zero");
        }
        if (kvcm.getHeartbeatFailureThreshold() <= 0
                || kvcm.getQueryFailureThreshold() <= 0
                || kvcm.getRecoverySuccessThreshold() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG kvcm health thresholds must be greater than zero");
        }
        validateLocalStandby(kvcm.getLocalStandby());
        serviceDiscovery.validate(kvcm.toEndpoint());
    }

    private void validateLocalStandby(LocalStandbyConfig localStandby) {
        if (localStandby == null) {
            return;
        }
        if (localStandby.getBlockSize() < 0
                || localStandby.getBlockSize() > Integer.MAX_VALUE
                || localStandby.getTtlMs() <= 0
                || localStandby.getMinimumTtlMs() <= 0
                || localStandby.getMinimumTtlMs() > localStandby.getTtlMs()
                || !Double.isFinite(localStandby.getTtlReductionStartRatio())
                || localStandby.getTtlReductionStartRatio() <= 0
                || localStandby.getTtlReductionStartRatio() >= 1
                || localStandby.getMaximumEntries() <= 0
                || !Double.isFinite(localStandby.getCapacityMultiplier())
                || localStandby.getCapacityMultiplier() < 1.0
                || localStandby.getAsyncQueueCapacity() <= 0
                || localStandby.getHashThreadCount() <= 0
                || localStandby.getHashQueueCapacity() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG kvcm.local_standby contains invalid values");
        }
    }

    private void validateOnlineOptimizer(
            OnlineOptimizerConfig optimizer,
            RoutingServiceDiscovery serviceDiscovery) {
        if (optimizer == null || !optimizer.isEnabled()) {
            return;
        }
        if (StringUtils.isBlank(optimizer.getInstanceGroup())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.instance_group must not be blank");
        }
        if (StringUtils.isBlank(optimizer.getInstanceId())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.instance_id must not be blank");
        }
        if (StringUtils.isBlank(optimizer.getPath())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.path must not be blank");
        }
        if (!optimizer.getPath().startsWith("/")) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.path must start with '/'");
        }
        if (optimizer.getPath().indexOf('?') >= 0 || optimizer.getPath().indexOf('#') >= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.path must not contain query or fragment");
        }
        if (optimizer.getDiscovery() != null
                && optimizer.getDiscovery().getPollIntervalMs() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.discovery.poll_interval_ms "
                            + "must be greater than zero");
        }
        if (optimizer.getRegisterTimeoutMs() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.register_timeout_ms must be greater than zero");
        }
        if (optimizer.getBlockSize() <= 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.block_size must be greater than zero");
        }
        if (optimizer.getLinearStep() < 0) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.linear_step must not be negative");
        }
        if (CollectionUtils.isEmpty(optimizer.getLocationSpecInfos())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos must not be empty");
        }
        if (CollectionUtils.isEmpty(optimizer.getLocationSpecGroups())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups must not be empty");
        }

        Set<String> specNames = new HashSet<>();
        for (OnlineOptimizerConfig.LocationSpecInfo spec : optimizer.getLocationSpecInfos()) {
            if (spec == null || StringUtils.isBlank(spec.getName())) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos entries require a name");
            }
            if (spec.getSize() <= 0) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos sizes must be greater than zero");
            }
            if (!specNames.add(spec.getName())) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos names must be unique: "
                                + spec.getName());
            }
        }

        Set<String> groupNames = new HashSet<>();
        Map<String, Set<String>> specsByGroup = new HashMap<>();
        for (OnlineOptimizerConfig.LocationSpecGroup group : optimizer.getLocationSpecGroups()) {
            if (group == null || StringUtils.isBlank(group.getName())) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups entries require a name");
            }
            if (CollectionUtils.isEmpty(group.getSpecNames())) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups spec_names must not be empty");
            }
            if (!groupNames.add(group.getName())) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups names must be unique: "
                                + group.getName());
            }
            Set<String> groupSpecNames = new HashSet<>();
            for (String specName : group.getSpecNames()) {
                if (StringUtils.isBlank(specName)) {
                    throw new IllegalArgumentException(
                            "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups spec_names must not be blank");
                }
                if (!specNames.contains(specName)) {
                    throw new IllegalArgumentException(
                            "MODEL_SERVICE_CONFIG online_optimizer location spec is not defined: "
                                    + specName);
                }
                if (!groupSpecNames.add(specName)) {
                    throw new IllegalArgumentException(
                            "MODEL_SERVICE_CONFIG online_optimizer location spec group contains duplicate spec: "
                                    + specName);
                }
            }
            specsByGroup.put(group.getName(), groupSpecNames);
        }
        OnlineOptimizerConfig.OptimizerStateInfo stateInfo = optimizer.getOptimizerStateInfo();
        if (stateInfo == null || StringUtils.isBlank(stateInfo.getFullLocationSpecGroupName())) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer.optimizer_state_info."
                            + "full_location_spec_group_name must not be blank");
        }
        String fullGroupName = stateInfo.getFullLocationSpecGroupName();
        String linearGroupName = stateInfo.getLinearLocationSpecGroupName();
        if (!groupNames.contains(fullGroupName)) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG online_optimizer full location spec group is not defined: "
                            + fullGroupName);
        }
        if (StringUtils.isNotBlank(linearGroupName)) {
            if (linearGroupName.equals(fullGroupName)) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer full and linear location spec groups must differ");
            }
            if (!groupNames.contains(linearGroupName)) {
                throw new IllegalArgumentException(
                        "MODEL_SERVICE_CONFIG online_optimizer linear location spec group is not defined: "
                            + linearGroupName);
            }
            for (String specName : specsByGroup.get(fullGroupName)) {
                if (specsByGroup.get(linearGroupName).contains(specName)) {
                    throw new IllegalArgumentException(
                            "MODEL_SERVICE_CONFIG online_optimizer full and linear location spec groups "
                                    + "must not share specs: " + specName);
                }
            }
        }
        serviceDiscovery.validate(optimizer.toEndpoint());
    }
}
