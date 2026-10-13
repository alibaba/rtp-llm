package org.flexlb.consistency;

import org.apache.curator.test.TestingServer;

import java.io.IOException;
import java.net.ServerSocket;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Standalone ZooKeeper process launcher for the HA case-test harness.
 *
 * <p>Lives on the flexlb-sync <b>test</b> classpath so it never pollutes the
 * production artifact. It uses only the JDK and curator-test's
 * {@link TestingServer} (no additional dependencies).</p>
 *
 * <p>Harness contract:
 * <pre>
 *   java -cp &lt;flexlb-sync test classpath&gt; \
 *       org.flexlb.consistency.ZkTestingServerLauncher --port &lt;p&gt;
 * </pre>
 * <ul>
 *   <li>After the embedded ZK is up, prints exactly one line
 *       {@code ZK_READY <connectString>} to stdout and flushes.</li>
 *   <li>{@code --port 0} (the default) picks a free port automatically; the
 *       ready line reports the actually bound port.</li>
 *   <li>Stays alive until it receives SIGTERM/SIGINT or its stdin reaches
 *       EOF (the harness may close the pipe to stop the server).</li>
 *   <li>Before exiting it prints {@code ZK_STOPPED} exactly once.</li>
 *   <li>Any startup failure prints the reason to stderr and exits with
 *       code 1.</li>
 * </ul></p>
 */
public final class ZkTestingServerLauncher {

    private ZkTestingServerLauncher() {
    }

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
        Runnable stopOnce = () -> {
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
