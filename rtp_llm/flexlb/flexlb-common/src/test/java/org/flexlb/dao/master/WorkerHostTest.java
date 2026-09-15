package org.flexlb.dao.master;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WorkerHostTest {
    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = "cn-hangzhou")
    void discoveryFactoryPreservesSiteAndDefaultPorts(String site) {
        // Keep the three-argument API used by the separately built internal VipServer adapter.
        List<WorkerHost> hosts = List.of("127.0.0.1").stream()
                .map(ip -> WorkerHost.of(ip, 8080, site))
                .collect(Collectors.toList());

        WorkerHost host = hosts.get(0);
        assertEquals("127.0.0.1:8080", host.getIpPort());
        assertEquals(8080, host.getPort());
        assertEquals(8081, host.getGrpcPort());
        assertEquals(8085, host.getHttpServerPort());
        assertEquals(site == null ? "" : site, host.getSite());
        assertEquals("", host.getGroup());
    }
}
