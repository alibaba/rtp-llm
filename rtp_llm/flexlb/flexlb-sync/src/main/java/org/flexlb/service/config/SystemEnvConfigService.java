package org.flexlb.service.config;

import com.alibaba.fastjson.JSON;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.domain.balance.WhaleMasterConfig;
import org.springframework.stereotype.Component;

@Getter
@Slf4j
@Component("configService")
public class SystemEnvConfigService implements ConfigService {

    private final WhaleMasterConfig whaleMasterConfig;

    public SystemEnvConfigService() {
        String lbConfigStr = System.getenv("WHALE_MASTER_CONFIG");
        log.warn("WHALE_MASTER_CONFIG = {}", lbConfigStr);
        WhaleMasterConfig config;
        if (lbConfigStr != null) {
            config = JSON.parseObject(lbConfigStr, WhaleMasterConfig.class);
        } else {
            config = new WhaleMasterConfig();
        }
        this.whaleMasterConfig = config;
    }

    @Override
    public WhaleMasterConfig loadBalanceConfig() {
        return whaleMasterConfig;
    }
}
