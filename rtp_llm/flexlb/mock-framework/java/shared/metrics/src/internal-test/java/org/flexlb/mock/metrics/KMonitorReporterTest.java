package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import com.taobao.kmonitor.impl.KMonitorConfig;
import java.util.Map;
import org.junit.jupiter.api.Test;

class KMonitorReporterTest {
    @Test void usesNativeWhaleConfigurationWithoutFlexlbOverrides() {
        long interval = KMonitorReporter.configure(Map.of("kmonitorServiceName", "engine-service",
            "kmonitorTenant", "tenant", "kmonitorSinkAddress", "10.1.2.3", "kmonitorPort", "4142",
            "kmonitorNormalSamplePeriod", "2"));
        assertEquals("engine-service", KMonitorConfig.getKMonitorServiceName());
        assertEquals("tenant", KMonitorConfig.getKMonitorTenantName());
        assertEquals("10.1.2.3:4142", KMonitorConfig.getSinkAddress());
        assertEquals(2000, interval);
        assertEquals(1000, KMonitorReporter.configure(Map.of("HIPPO_SLAVE_IP", "10.2.3.4")));
        assertEquals("10.2.3.4:4141", KMonitorConfig.getSinkAddress());
        assertThrows(IllegalArgumentException.class, () -> KMonitorReporter.configure(Map.of("kmonitorPort", "0")));
    }
}
