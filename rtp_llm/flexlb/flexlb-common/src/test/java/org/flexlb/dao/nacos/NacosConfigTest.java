package org.flexlb.dao.nacos;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class NacosConfigTest {

    @Test
    void usesDefaultGroupAndNamespace() {
        NacosConfig config = new NacosConfig(
                "127.0.0.1:8848",
                "flexlb-hongyi-test-v1-flexlb-standalone",
                null,
                null);

        assertThat(config.getDataId()).isEqualTo("flexlb-hongyi-test-v1-flexlb-standalone");
        assertThat(config.getGroup()).isEqualTo("DEFAULT_GROUP");
        assertThat(config.getNamespace()).isEmpty();
    }

    @Test
    void usesExplicitDataIdGroupAndNamespace() {
        NacosConfig config = new NacosConfig(
                "nacos.example:8848",
                "explicit-data-id",
                "FLEXLB_GROUP",
                "production");

        assertThat(config)
                .isEqualTo(new NacosConfig(
                        "nacos.example:8848",
                        "explicit-data-id",
                        "FLEXLB_GROUP",
                        "production"));
    }
}
