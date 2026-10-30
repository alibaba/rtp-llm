package org.flexlb.mock.master;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.InetSocketAddress;
import java.nio.file.*;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import org.apache.zookeeper.server.NIOServerCnxnFactory;
import org.apache.zookeeper.server.ZooKeeperServer;

/** A real, runner-owned standalone ZooKeeper; independent of every Master lifecycle. */
public final class ZookeeperNode {
    public static void main(String[] args) {
        try { run(); }
        catch (Throwable error) {
            error.printStackTrace(System.err);
            System.exit(1); // Native libraries may leave non-daemon threads after failed initialization.
        }
    }
    private static void run() throws Exception {
        Path directory = Path.of(System.getenv("MOCK_OUTPUT_DIR"));
        Path data = Files.createDirectories(directory.resolve("data"));
        var server = new ZooKeeperServer(data.toFile(), data.toFile(), 2000);
        var connections = new NIOServerCnxnFactory();
        try {
            connections.configure(new InetSocketAddress("127.0.0.1", 0), 64);
            connections.startup(server);
            Runtime.getRuntime().addShutdownHook(new Thread(connections::shutdown, "mock-zk-close"));
            Path ready = Path.of(System.getenv("MOCK_READY_FILE"));
            Path temporary = Files.createTempFile(ready.getParent(), ".ready-", ".json");
            new ObjectMapper().writeValue(temporary.toFile(), Map.of("connect", "127.0.0.1:" + connections.getLocalPort()));
            Files.move(temporary, ready, StandardCopyOption.ATOMIC_MOVE);
            new CountDownLatch(1).await();
        } finally {
            connections.shutdown();
            server.shutdown();
        }
    }
}
