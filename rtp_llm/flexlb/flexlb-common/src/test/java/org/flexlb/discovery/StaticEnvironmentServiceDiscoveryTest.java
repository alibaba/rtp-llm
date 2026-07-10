package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ExtendWith(SystemStubsExtension.class)
class StaticEnvironmentServiceDiscoveryTest {

    @SystemStub
    private final EnvironmentVariables environment = new EnvironmentVariables();

    private final StaticEnvironmentServiceDiscovery discovery =
            StaticEnvironmentServiceDiscovery.getInstance();

    @Test
    void readsHostsFromPortableEnvironmentVariable() {
        environment.set(
                "FLEXLB_DISCOVERY_STATIC_HOSTS_COM_EXAMPLE_PREFILL",
                "127.0.0.1:8080, 127.0.0.2:8081");

        List<WorkerHost> hosts = discovery.getHosts("com.example.prefill");

        assertEquals(2, hosts.size());
        assertEquals("127.0.0.1:8080", hosts.get(0).getIpPort());
        assertEquals("127.0.0.2:8081", hosts.get(1).getIpPort());
    }

    @Test
    void keepsLegacyEnvironmentVariableCompatible() {
        environment.set("DOMAIN_ADDRESS:com.example.decode", "127.0.0.3:8082");

        List<WorkerHost> hosts = discovery.getHosts("com.example.decode");

        assertEquals(1, hosts.size());
        assertEquals("127.0.0.3:8082", hosts.get(0).getIpPort());
    }

    @Test
    void normalizesAddressForEnvironmentVariableName() {
        assertEquals(
                "FLEXLB_DISCOVERY_STATIC_HOSTS_V_63CBCBA2",
                StaticEnvironmentServiceDiscovery.environmentVariableName("v-63cbcba2"));
    }
}
