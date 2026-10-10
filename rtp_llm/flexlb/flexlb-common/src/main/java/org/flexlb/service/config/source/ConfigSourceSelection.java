package org.flexlb.service.config.source;

import org.apache.commons.lang3.StringUtils;

import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;

enum ConfigSourceSelection {
    UNICONFIG,
    NACOS,
    ENVIRONMENT;

    static ConfigSourceSelection fromEnvironment() {
        if (Boolean.parseBoolean(StringUtils.trimToEmpty(System.getenv("FLEXLB_UNICONF_ENABLE")))) {
            return UNICONFIG;
        }
        if (StringUtils.isNotBlank(System.getenv(NACOS_SERVER_ADDR))) {
            return NACOS;
        }
        return ENVIRONMENT;
    }
}
