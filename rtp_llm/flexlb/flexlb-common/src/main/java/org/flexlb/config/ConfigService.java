package org.flexlb.config;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

@Getter
@Slf4j
@Component
public class ConfigService {

    private final FlexlbConfig flexlbConfig;

    public ConfigService() {
        String lbConfigStr = System.getenv("FLEXLB_CONFIG");
        log.warn("FLEXLB_CONFIG = {}", lbConfigStr);
        FlexlbConfig config;
        if (lbConfigStr != null) {
            config = JsonUtils.toObject(lbConfigStr, FlexlbConfig.class);
        } else {
            config = new FlexlbConfig();
        }

        // Per-field env-var overrides take precedence over the JSON blob (e.g. ENABLE_QUEUEING=true
        // overrides whatever enableQueueing was in FLEXLB_CONFIG). No prefix — matches the
        // historical FLEXLB_CONFIG contract operators rely on.
        EnvConfigOverrides.apply(config, "");

        this.flexlbConfig = config;
    }

    public FlexlbConfig loadBalanceConfig() {
        return flexlbConfig;
    }
}
