package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the empty-vs-throw contract of {@link NoOpServiceDiscovery#getHosts(String)}: an empty
 * fleet is legal and returns an empty list, while a malformed {@code DOMAIN_ADDRESS} value is a
 * lookup failure and throws. Callers act on "empty" as authoritative (embedding liveness marks a
 * whole fleet dead on it), so the two must never collapse into each other.
 */
class NoOpServiceDiscoveryTest {

    private static final String ADDRESS = "my.service";

    private static Map<String, String> env(String hostsConfig) {
        Map<String, String> env = new HashMap<>();
        env.put("DOMAIN_ADDRESS:" + ADDRESS, hostsConfig);
        return env;
    }

    private static List<WorkerHost> resolve(String hostsConfig) {
        return NoOpServiceDiscovery.getInstance().getHosts(ADDRESS, env(hostsConfig));
    }

    @Test
    void parsesValidHostList() {
        List<WorkerHost> hosts = resolve("10.0.0.1:8080, 10.0.0.2:9090");

        assertEquals(2, hosts.size());
        assertEquals("10.0.0.1", hosts.get(0).getIp());
        assertEquals(8080, hosts.get(0).getPort());
        assertEquals("10.0.0.2", hosts.get(1).getIp());
        assertEquals(9090, hosts.get(1).getPort());
    }

    @Test
    void missingPortThrowsIllegalArgument() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> resolve("10.0.0.1"));
        assertTrue(e.getMessage().contains(ADDRESS),
                "error must name the address the malformed value belongs to");
    }

    @Test
    void garbageSeparatorThrowsIllegalArgument() {
        assertThrows(IllegalArgumentException.class, () -> resolve("10.0.0.1:8080;10.0.0.2:9090"));
    }

    @Test
    void nonNumericPortThrowsIllegalArgument() {
        assertThrows(IllegalArgumentException.class, () -> resolve("10.0.0.1:http"));
    }

    @Test
    void blankEntriesAreFilteredNotRejected() {
        assertEquals(2, resolve("10.0.0.1:8080,, ,10.0.0.2:9090").size());
    }

    @Test
    void allBlankValueMeansEmptyFleetNotFailure() {
        assertTrue(resolve(" , ").isEmpty());
    }

    @Test
    void missingEnvKeyMeansEmptyFleetNotFailure() {
        assertTrue(NoOpServiceDiscovery.getInstance().getHosts(ADDRESS, Map.of()).isEmpty());
    }

    @Test
    void blankAddressMeansEmptyFleetNotFailure() {
        assertTrue(NoOpServiceDiscovery.getInstance().getHosts(" ", Map.of()).isEmpty());
    }

    @Test
    void listenSwallowsAMalformedConfigAndPushesNothing() {
        // getHosts throws on this input (missing port); listen is a registration, not a lookup,
        // so the same failure must be logged and swallowed — and no host list may be pushed.
        AtomicReference<List<WorkerHost>> pushed = new AtomicReference<>();
        assertDoesNotThrow(() -> NoOpServiceDiscovery.getInstance()
                .listen(ADDRESS, pushed::set, env("10.0.0.1")));
        assertNull(pushed.get(), "a failed resolve must not be pushed as if it were a fleet");
    }

    @Test
    void listenPushesHostsAndDoesNotPropagateLookupFailure() {
        // Registration is not a lookup: listen() must not propagate a resolution failure, so
        // callers registering a listener need no defensive wrapper around it.
        AtomicReference<List<WorkerHost>> pushed = new AtomicReference<>();
        NoOpServiceDiscovery.getInstance().listen("no.such.address.for.test", pushed::set);
        assertTrue(pushed.get().isEmpty(), "an unresolvable address pushes an empty fleet");

        AtomicReference<List<WorkerHost>> notPushed = new AtomicReference<>();
        NoOpServiceDiscovery.getInstance().listen("blank.address.test", null);
        assertNull(notPushed.get(), "a null listener is a no-op");
    }
}
