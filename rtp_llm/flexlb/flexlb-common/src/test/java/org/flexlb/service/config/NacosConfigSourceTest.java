package org.flexlb.service.config;

import com.alibaba.nacos.api.config.listener.Listener;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.nacos.NacosConfig;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.constant.NacosConfigConstants.HIPPO_ROLE;
import static org.flexlb.constant.NacosConfigConstants.NACOS_DATA_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_NAMESPACE;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class NacosConfigSourceTest {

    @Test
    void isDisabledWhenNacosAddressIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables()
                .remove(NACOS_SERVER_ADDR)
                .execute(NacosConfigSource::new);

        source.initialize();
        ConfigService configService = new ConfigService();

        assertThat(source.priority()).isEqualTo(2);
        assertThat(configService.loadBalanceConfig().getMaxRetryCount()).isZero();
        configService.close();
    }

    @Test
    void failsFastWhenDataIdCannotBeResolved() {
        EnvironmentVariables environment = new EnvironmentVariables(NACOS_SERVER_ADDR, "127.0.0.1:8848")
                .remove(NACOS_DATA_ID)
                .remove(HIPPO_ROLE);

        assertThatThrownBy(() -> environment.execute(NacosConfigSource::new))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(HIPPO_ROLE);
    }

    @Test
    void usesHippoRoleWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                HIPPO_ROLE, "flexlb-hongyi-test-v1-flexlb-standalone")
                .remove(NACOS_DATA_ID)
                .execute(NacosConfigSource::new);

        assertThat(source)
                .extracting("config")
                .isEqualTo(new NacosConfig(
                        "127.0.0.1:8848",
                        "flexlb-hongyi-test-v1-flexlb-standalone",
                        null,
                        null));
    }

    @Test
    void loadsListensAndClosesNacosConfig() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfigAndSignListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L),
                listenerCaptor.capture()))
                .thenReturn("{\"maxRetryCount\":9}");
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        ConfigService configService = new ConfigService();

        assertThat(configService.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
        listenerCaptor.getValue().receiveConfigInfo("{\"maxRetryCount\":10}");
        configService.close();

        assertThat(configService.loadBalanceConfig().getMaxRetryCount()).isEqualTo(10);
        verify(client).removeListener(
                "flexlb-test",
                "FLEXLB_GROUP",
                listenerCaptor.getValue());
        verify(client).shutDown();
    }

    @Test
    void shutsDownClientWhenRemovingListenerFails() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        when(client.getConfigAndSignListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L),
                org.mockito.ArgumentMatchers.any(Listener.class)))
                .thenReturn("{\"maxRetryCount\":9}");
        NacosConfigSource source = createSource(client, "");
        source.initialize();
        ConfigService configService = new ConfigService();
        doThrow(new RuntimeException("remove failed"))
                .when(client)
                .removeListener(
                        org.mockito.ArgumentMatchers.eq("flexlb-test"),
                        org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                        org.mockito.ArgumentMatchers.any(Listener.class));

        assertThatThrownBy(source::close).hasMessage("remove failed");

        verify(client).shutDown();
        configService.close();
    }

    private NacosConfigSource createSource(
            com.alibaba.nacos.api.config.ConfigService client,
            String namespace) throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                NACOS_DATA_ID, "flexlb-test",
                NACOS_GROUP, "FLEXLB_GROUP",
                NACOS_NAMESPACE, namespace)
                .execute(NacosConfigSource::new);
        ReflectionTestUtils.setField(source, "client", client);
        return source;
    }
}
