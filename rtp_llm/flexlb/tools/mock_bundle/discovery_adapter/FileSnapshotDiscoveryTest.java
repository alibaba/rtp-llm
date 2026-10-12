package org.flexlb.mockdiscovery;

import org.flexlb.config.ServiceDiscoveryConfiguration;
import org.flexlb.discovery.ServiceDiscovery;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.MapPropertySource;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class FileSnapshotDiscoveryTest {
    @TempDir Path dir;
    @Test void pinnedMasterSelectsAdapterAndReloadsAddRemoveWithoutRestart() throws Exception {
        Path file = dir.resolve("discovery.json");
        Files.writeString(file, "{\"p\":[\"127.0.0.1:7050\"],\"d\":[]}");
        try (var context = new AnnotationConfigApplicationContext()) {
            context.getEnvironment().getPropertySources().addFirst(new MapPropertySource("mock", Map.of(
                    "mock.discovery.file", file.toString())));
            context.register(ServiceDiscoveryConfiguration.class, FileSnapshotDiscovery.class);
            context.refresh();
            var service = context.getBean(ServiceDiscovery.class);
            assertInstanceOf(FileSnapshotDiscovery.class, service);
            assertEquals(1, service.getHosts("p").size());
            Files.writeString(file, "{\"p\":[\"127.0.0.1:7050\",\"127.0.0.1:7051\"],\"d\":[]}");
            assertEquals(2, service.getHosts("p").size());
            Files.writeString(file, "broken");
            assertEquals(2, service.getHosts("p").size());
            Files.writeString(file, "{\"p\":[],\"d\":[]}");
            assertTrue(service.getHosts("p").isEmpty());
        }
    }
    @Test void disabledAdapterPreservesDefaultAndInitialBadFileFails() {
        try (var context = new AnnotationConfigApplicationContext(ServiceDiscoveryConfiguration.class,
                FileSnapshotDiscovery.class)) {
            assertFalse(context.getBean(ServiceDiscovery.class) instanceof FileSnapshotDiscovery);
        }
        assertThrows(IllegalStateException.class,
                () -> new FileSnapshotDiscovery(dir.resolve("missing").toString()).getHosts("p"));
    }
}
