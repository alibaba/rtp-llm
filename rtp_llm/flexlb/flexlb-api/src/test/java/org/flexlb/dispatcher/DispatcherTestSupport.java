package org.flexlb.dispatcher;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.web.reactive.function.client.WebClient;

import java.net.URI;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

final class DispatcherTestSupport {
    static DispatcherMetricsReporter noopMetrics() {
        return new DispatcherMetricsReporter(NoOpFlexMonitor.getInstance());
    }

    static ConfigService configService(FlexlbConfig config) {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    static FePool fePool(List<String> urls) {
        FePool pool = new FePool(mock(ServiceDiscovery.class), WebClient.create(), new DispatchConfig(), noopMetrics());
        pool.update(urls.stream().map(URI::create).map(uri -> WorkerHost.of(uri.getHost(), uri.getPort())).toList());
        return pool;
    }
}
