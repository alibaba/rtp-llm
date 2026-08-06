package org.flexlb.config;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.config.YamlPropertiesFactoryBean;
import org.springframework.core.io.ClassPathResource;

import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;

class CodecDefaultConfigurationTest {

    @Test
    void defaultsCodecBufferLimitToTenMiB() {
        YamlPropertiesFactoryBean yaml = new YamlPropertiesFactoryBean();
        yaml.setResources(new ClassPathResource("application.yml"));
        Properties properties = yaml.getObject();

        assertThat(properties)
                .containsEntry("spring.codec.max-in-memory-size", "${MAX_IN_MEMORY_SIZE:10MB}");
    }
}
