package org.flexlb.service.config.source;

import com.alibaba.nacos.api.NacosFactory;
import com.alibaba.nacos.api.PropertyKeyConst;
import com.alibaba.nacos.api.config.listener.Listener;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.dao.nacos.NacosConfig;
import org.flexlb.service.config.ConfigSource;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Properties;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

import static org.flexlb.constant.NacosConfigConstants.NACOS_DATA_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_NAMESPACE;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;

@Slf4j
@Component
final class NacosConfigSource implements ConfigSource {

    private static final long CONFIG_READ_TIMEOUT_MS = 3000;
    private static final int PRIORITY = 2;

    private final NacosConfig config;
    private com.alibaba.nacos.api.config.ConfigService client;
    private Listener listener;
    private volatile Consumer<String> updateListener;
    private volatile String configContent;

    NacosConfigSource(DeploymentIdentity deploymentIdentity) {
        String serverAddr = StringUtils.trimToNull(System.getenv(NACOS_SERVER_ADDR));
        if (serverAddr == null) {
            this.config = null;
            return;
        }

        String dataId = StringUtils.trimToNull(System.getenv(NACOS_DATA_ID));
        if (dataId == null) {
            dataId = deploymentIdentity.getDeploymentId();
        }
        String group = StringUtils.trimToNull(System.getenv(NACOS_GROUP));
        String namespace = StringUtils.trimToNull(System.getenv(NACOS_NAMESPACE));
        this.config = new NacosConfig(serverAddr, dataId, group, namespace);
    }

    @Override
    public String name() {
        return "Nacos";
    }

    @Override
    public int priority() {
        return PRIORITY;
    }

    @PostConstruct
    void initialize() {
        if (config == null) {
            log.info("Nacos configuration source is disabled");
            return;
        }
        try {
            if (client == null) {
                client = createClient(config);
            }
            listener = createListener();
            configContent = client.getConfig(config.getDataId(), config.getGroup(), CONFIG_READ_TIMEOUT_MS);
            client.addListener(config.getDataId(), config.getGroup(), listener);
            ConfigService.register(this);
        } catch (Exception e) {
            try {
                close();
            } catch (Exception closeException) {
                e.addSuppressed(closeException);
            }
            throw new IllegalStateException("Failed to initialize Nacos configuration source", e);
        }
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {
        this.updateListener = listener;
    }

    @Override
    public String load() {
        return configContent;
    }

    @Override
    public void close() throws Exception {
        if (client == null) {
            return;
        }
        try {
            if (listener != null) {
                client.removeListener(config.getDataId(), config.getGroup(), listener);
            }
        } finally {
            client.shutDown();
        }
    }

    private com.alibaba.nacos.api.config.ConfigService createClient(NacosConfig config) throws Exception {
        Properties properties = new Properties();
        properties.put(PropertyKeyConst.SERVER_ADDR, config.getServerAddr());
        properties.put(PropertyKeyConst.IS_USE_CLOUD_NAMESPACE_PARSING, Boolean.FALSE.toString());
        properties.put(PropertyKeyConst.NAMESPACE, config.getNamespace());
        return NacosFactory.createConfigService(properties);
    }

    private Listener createListener() {
        return new Listener() {
            @Override
            public Executor getExecutor() {
                return null;
            }

            @Override
            public void receiveConfigInfo(String configInfo) {
                configContent = configInfo;
                Consumer<String> listener = updateListener;
                if (listener != null) {
                    listener.accept(configInfo);
                }
            }
        };
    }

}
