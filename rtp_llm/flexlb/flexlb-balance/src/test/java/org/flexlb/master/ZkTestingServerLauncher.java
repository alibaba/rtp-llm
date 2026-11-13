package org.flexlb.master;

import org.apache.curator.test.TestingServer;

import java.io.IOException;
import java.net.ServerSocket;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Starts an embedded ZooKeeper process for HA test harnesses and exposes its connection address. It
 * lives only on the test classpath.
 */
public final class ZkTestingServerLauncher {

    private ZkTestingServerLauncher() {}

    public static void main(String[] args) {
        int port = parsePort(args);
        TestingServer server;
        try {
            // Resolve port 0 here: TestingServer.getConnectString() reports the
            // port from its InstanceSpec, so pre-allocate a concrete free port
            // instead of relying on ZooKeeper's own port-0 auto-binding.
            if (port == 0) {
                port = findFreePort();
            }
            // TestingServer owns a temp data dir and starts synchronously.
            server = new TestingServer(port, true);
        } catch (Exception e) {
            System.err.println("ZK_START_FAILED " + e);
            System.exit(1);
            return;
        }

        AtomicBoolean stopped = new AtomicBoolean(false);
        Runnable stopOnce =
                () -> {
                    if (stopped.compareAndSet(false, true)) {
                        try {
                            server.stop();
                        } catch (Exception ignored) {
                            // best effort
                        }
                        System.out.println("ZK_STOPPED");
                        System.out.flush();
                    }
                };
        Runtime.getRuntime().addShutdownHook(new Thread(stopOnce, "zk-launcher-stop"));

        System.out.println("ZK_READY " + server.getConnectString());
        System.out.flush();

        try {
            // Block until stdin EOF; SIGTERM is handled by the shutdown hook.
            while (System.in.read() != -1) {
                // discard stdin bytes; harness may keep the pipe open silently
            }
        } catch (IOException e) {
            // stdin unavailable (e.g. closed/redirected): fall through to stop
        }
        stopOnce.run();
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private static int parsePort(String[] args) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--port".equals(args[i])) {
                try {
                    return Integer.parseInt(args[i + 1]);
                } catch (NumberFormatException e) {
                    System.err.println("invalid --port value: " + args[i + 1]);
                    System.exit(2);
                }
            }
        }
        return 0;
    }
}
