package org.flexlb.config;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * @author jingyu
 */
@ConfigurationProperties(prefix = "whale.api.backend")
@Component
@Data
@Slf4j
public class WhaleApiBackendConfig {
    private int requestTimeoutMillis = 500;
    private int handShakeTimeoutMillis = 500;
    private int responseTimeoutSeconds = 120;
    private int nettyMaxChunkSize = 8092;
    private boolean isCompression = false;
    private int connectMaxTryTimes = 3;
    private int eventExecuteThreads = 10 * Runtime.getRuntime().availableProcessors();
}
