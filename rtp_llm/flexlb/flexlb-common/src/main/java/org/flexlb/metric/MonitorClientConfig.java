package org.flexlb.metric;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import com.taobao.kmonitor.ImmutableMetricTags;
import com.taobao.kmonitor.KMonitor;
import com.taobao.kmonitor.KMonitorFactory;
import com.taobao.kmonitor.core.MetricsTags;
import com.taobao.kmonitor.impl.KMonitorConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MonitorClientConfig {

    @Bean
    public KMonitor initMonitorClient() {
        try {
            KMonitorConfig.setKMonitorServiceName("whale-lb");
            KMonitorConfig.setKMonitorTenantName("default");
            KMonitorFactory.start();

            // 设置全局tags
            Map<String, String> tags = new HashMap<>();
            Optional.ofNullable(System.getenv("HIPPO_ROLE")).ifPresent(hippoRole -> tags.put("HIPPO_ROLE", hippoRole));
            Optional.ofNullable(System.getenv("HIPPO_APP")).ifPresent(hippoApp -> tags.put("HIPPO_APP", hippoApp));
            Optional.ofNullable(System.getenv("BIZ_NAME")).ifPresent(hippoApp -> tags.put("BIZ_NAME", hippoApp));

            KMonitorFactory.addGlobalTags(new ImmutableMetricTags(tags));
            MetricsTags globalTags = KMonitorFactory.getGlobalTags();
            log.info("init kmonitor client: globalTags={}", globalTags);

            return KMonitorFactory.getKMonitor("whale-lb");
        } catch (Exception e) {
            log.error("init monitor client error, msg={}", e.getMessage());
            return new EmptyKmonitor();
        }
    }

}
