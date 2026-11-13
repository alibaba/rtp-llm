package org.flexlb.config.loading.source;

import org.apache.commons.lang3.StringUtils;

import static org.flexlb.config.loading.source.NacosConfigConstants.NACOS_SERVER_ADDR;

/**
 * Orders candidate configuration sources and retains the selected source and document.
 */
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
