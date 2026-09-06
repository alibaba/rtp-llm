package org.flexlb.consistency;

import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.leader.LeaderSelector;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.apache.curator.test.TestingServer;
import org.apache.curator.utils.CloseableUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.Timeout;
import org.springframework.core.env.Environment;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.net.ServerSocket;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tier-2 HA case tests: a real ZooKeeper ensemble (curator-test
 * {@link TestingServer}) drives the production
 * {@link ZookeeperMasterElectService} election code.
 *
 * <p>Construction note: {@code ZookeeperMasterElectService}'s constructor runs
 * {@code init()}, which reads the consistency policy from {@link ConfigService}.
 * The test supplies the default {@link FlexlbConfig}, so {@code init()} returns
 * before touching deployment identity or ZooKeeper. The test then uses
 * the production {@code @Setter} hooks (including the package-private
 * {@code setClient}/{@code setLeaderSelector} reachable from this same
 * package) to bind a real Curator client and a real {@link LeaderSelector}
 * fighting over the production lock path
 * {@code /master_lb_leader/{roleId}} under the production
 * {@code whale-master} namespace.</p>
 *
 * <p>Covered production semantics (brief p2 / p6):
 * <ul>
 *   <li>exactly one of two racers holds the ephemeral lock
 *       ({@code isMaster} true for exactly one);</li>
 *   <li>the leader's ZK session dying (Mode-1 process death) deletes the
 *       ephemeral node immediately and wakes the queued follower
 *       ({@code autoRequeue()});</li>
 *   <li>a restarted node rejoins as a follower while the survivor keeps
 *       leadership — no split brain at any sampled instant.</li>
 * </ul></p>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ZkLeaderElectionTest {

    private static final String ROLE_ID = "ha-casetest-role";
    private static final String LEADER_PATH = "/master_lb_leader/" + ROLE_ID;
    private static final String MASTER_NAMESPACE = "whale-master";
    private static final int SESSION_TIMEOUT_MS = 10_000;
    private static final long AWAIT_TIMEOUT_MS = 20_000;
    private static final long STABLE_WINDOW_MS = 2_000;

    private TestingServer zkServer;
    private final List<AutoCloseable> perTestResources = new CopyOnWriteArrayList<>();

    @BeforeAll
    void startEmbeddedZk() throws Exception {
        zkServer = new TestingServer(freePort(), true);
    }

    @AfterAll
    void stopEmbeddedZk() {
        CloseableUtils.closeQuietly(zkServer);
    }

    @AfterEach
    void closePerTestResources() {
        for (AutoCloseable resource : perTestResources) {
            try {
                resource.close();
            } catch (Exception ignored) {
                // best-effort teardown
            }
        }
        perTestResources.clear();
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    @DisplayName("two electors racing for the ephemeral lock elect exactly one master")
    void twoElectorsElectExactlyOneMaster() {
        try (SplitBrainSampler sampler = new SplitBrainSampler()) {
            BoundElector a = startElector("127.0.0.1");
            BoundElector b = startElector("127.0.0.2");
            sampler.watch(a);
            sampler.watch(b);

            awaitUniqueMaster(a, b);
            BoundElector master = masterOf(a, b);
            BoundElector follower = master == a ? b : a;

            assertTrue(master.service().isMaster(), "elected leader must report isMaster=true");
            assertFalse(follower.service().isMaster(), "loser must report isMaster=false");

            // Leader identity: the leader sees itself; the follower sees the
            // leader after one view refresh (production refreshes every 5s).
            follower.service().updateLatestMaster();
            assertEquals(master.localIp(), master.service().getMasterHostIp(false),
                    "leader must answer getMasterHostIp with its own identity");
            assertEquals(master.localIp(), follower.service().getMasterHostIp(false),
                    "follower must answer getMasterHostIp with the leader's identity");

            sampler.assertNeverSplitBrain();
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    @DisplayName("leader ZK session drop deletes the ephemeral node and the follower takes over within a bounded timeout")
    void leaderSessionDropFollowerTakesOver() {
        try (SplitBrainSampler sampler = new SplitBrainSampler()) {
            BoundElector a = startElector("127.0.0.1");
            BoundElector b = startElector("127.0.0.2");
            sampler.watch(a);
            sampler.watch(b);

            awaitUniqueMaster(a, b);
            BoundElector leader = masterOf(a, b);
            BoundElector follower = leader == a ? b : a;

            // Simulate Mode-1 process death: closing the Curator client drops
            // the TCP session; ZooKeeper deletes the ephemeral node at once,
            // which is the fastest failover path (brief p2 trigger 1).
            killElector(leader);

            awaitUniqueMaster(follower);
            assertTrue(follower.service().isMaster(),
                    "follower must be woken up by autoRequeue and take leadership");
            assertFalse(leader.service().isMaster(),
                    "dead leader must not keep isMaster=true after its session died");

            sampler.assertNeverSplitBrain();
        }
    }

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    @DisplayName("restarted elector rejoins as follower while the survivor keeps leadership (no split brain)")
    void restartedLeaderRejoinsAsFollowerWithoutSplitBrain() {
        try (SplitBrainSampler sampler = new SplitBrainSampler()) {
            BoundElector a = startElector("127.0.0.1");
            BoundElector b = startElector("127.0.0.2");
            sampler.watch(a);
            sampler.watch(b);

            awaitUniqueMaster(a, b);
            BoundElector firstLeader = masterOf(a, b);
            BoundElector survivor = firstLeader == a ? b : a;

            // Hard-kill the leader, wait for the survivor to take over.
            killElector(firstLeader);
            awaitUniqueMaster(survivor);
            assertTrue(survivor.service().isMaster(),
                    "survivor must take over after the leader dies");

            // Restart the dead node: a fresh client + selector bound to the
            // same service instance, exactly like a restarted process.
            BoundElector rejoined = bindAndStart(firstLeader.service(), firstLeader.localIp());
            sampler.watch(rejoined);

            // The restarted node must rejoin as a FOLLOWER and stay there for
            // a stable window; the survivor keeps the lock.
            awaitStableMasterState(survivor, rejoined);
            assertFalse(rejoined.service().isMaster(),
                    "restarted node must rejoin as follower, not steal leadership back");
            assertTrue(survivor.service().isMaster(),
                    "survivor must keep leadership across the restart");

            // The restarted follower resolves the current leader's identity.
            rejoined.service().updateLatestMaster();
            assertEquals(survivor.localIp(), rejoined.service().getMasterHostIp(false));

            sampler.assertNeverSplitBrain();
        }
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    /** An elector service bound to a live curator client + leader selector. */
    private record BoundElector(ZookeeperMasterElectService service,
                                CuratorFramework client,
                                LeaderSelector selector,
                                String localIp) implements AutoCloseable {
        @Override
        public void close() {
            CloseableUtils.closeQuietly(selector);
            CloseableUtils.closeQuietly(client);
        }
    }

    private BoundElector startElector(String localIp) {
        return bindAndStart(newElectService(localIp), localIp);
    }

    /**
     * Builds the production service with mocked auxiliary dependencies. The
     * constructor's {@code init()} no-ops because the supplied default
     * configuration has no ZooKeeper consistency policy.
     */
    private ZookeeperMasterElectService newElectService(String localIp) {
        GeneralHttpNettyService httpService = mock(GeneralHttpNettyService.class);
        when(httpService.request(any(), any(), anyString(), any()))
                .thenReturn(Mono.empty());
        EngineHealthReporter healthReporter = mock(EngineHealthReporter.class);
        Environment environment = mock(Environment.class);

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        DeploymentIdentity deploymentIdentity = mock(DeploymentIdentity.class);
        ZookeeperMasterElectService service = new ZookeeperMasterElectService(
                httpService, healthReporter, environment, configService, deploymentIdentity);
        service.setRoleId(ROLE_ID);
        service.setLocalIp(localIp);
        service.setPort(18_080);
        return service;
    }

    /** Creates a fresh curator client + auto-requeue selector and starts election. */
    private BoundElector bindAndStart(ZookeeperMasterElectService service, String localIp) {
        CuratorFramework client = CuratorFrameworkFactory.builder()
                .namespace(MASTER_NAMESPACE)
                .connectString(zkServer.getConnectString())
                .sessionTimeoutMs(SESSION_TIMEOUT_MS)
                .connectionTimeoutMs(SESSION_TIMEOUT_MS)
                .retryPolicy(new ExponentialBackoffRetry(1000, 3))
                .build();
        client.start();
        assertDoesNotThrow(() -> client.blockUntilConnected(10, TimeUnit.SECONDS),
                "curator client must connect to the embedded ZK");

        LeaderSelector selector = new LeaderSelector(client, LEADER_PATH, service);
        selector.setId(localIp);
        // Production calls autoRequeue: losers re-enter the queue automatically.
        selector.autoRequeue();
        service.setClient(client);
        service.setLeaderSelector(selector);

        BoundElector bound = new BoundElector(service, client, selector, localIp);
        perTestResources.add(bound);
        service.start();
        return bound;
    }

    /** Mode-1 style hard death: selector first (stops its worker threads), then client. */
    private void killElector(BoundElector elector) {
        perTestResources.remove(elector);
        elector.close();
    }

    private static BoundElector masterOf(BoundElector... electors) {
        return Arrays.stream(electors)
                .filter(elector -> elector.service().isMaster())
                .findFirst()
                .orElseThrow(() -> new AssertionError("no master among electors"));
    }

    private void awaitUniqueMaster(BoundElector... electors) {
        long deadline = System.currentTimeMillis() + AWAIT_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            long masters = Arrays.stream(electors)
                    .filter(elector -> elector.service().isMaster())
                    .count();
            if (masters == 1) {
                return;
            }
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                fail("interrupted while waiting for a unique master");
            }
        }
        fail("no unique master within " + AWAIT_TIMEOUT_MS + "ms: " + describe(electors));
    }

    private void awaitStableMasterState(BoundElector expectedMaster, BoundElector expectedFollower) {
        long deadline = System.currentTimeMillis() + AWAIT_TIMEOUT_MS;
        long stableSince = -1L;
        while (System.currentTimeMillis() < deadline) {
            boolean state = expectedMaster.service().isMaster()
                    && !expectedFollower.service().isMaster();
            long now = System.currentTimeMillis();
            if (state) {
                if (stableSince < 0) {
                    stableSince = now;
                }
                if (now - stableSince >= STABLE_WINDOW_MS) {
                    return;
                }
            } else {
                stableSince = -1;
            }
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                fail("interrupted while waiting for a stable master state");
            }
        }
        fail("master state never stabilized: expected master=" + describe(expectedMaster)
                + ", expected follower=" + describe(expectedFollower));
    }

    private static String describe(BoundElector... electors) {
        StringBuilder sb = new StringBuilder("[");
        for (BoundElector elector : electors) {
            sb.append(elector.localIp()).append(" isMaster=")
                    .append(elector.service().isMaster()).append(", ");
        }
        return sb.append("]").toString();
    }

    /**
     * Samples {@code isMaster()} of every watched elector and records any
     * instant at which more than one elector claims leadership (split brain).
     */
    private final class SplitBrainSampler implements AutoCloseable {
        private final List<BoundElector> watched = new CopyOnWriteArrayList<>();
        private final AtomicBoolean splitBrain = new AtomicBoolean(false);
        private final ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor(
                runnable -> {
                    Thread thread = new Thread(runnable, "split-brain-sampler");
                    thread.setDaemon(true);
                    return thread;
                });

        SplitBrainSampler() {
            executor.scheduleAtFixedRate(this::sample, 0, 20, TimeUnit.MILLISECONDS);
        }

        void watch(BoundElector elector) {
            watched.add(elector);
        }

        private void sample() {
            long masters = watched.stream()
                    .filter(elector -> elector.service().isMaster())
                    .count();
            if (masters > 1) {
                splitBrain.set(true);
            }
        }

        void assertNeverSplitBrain() {
            assertFalse(splitBrain.get(),
                    "split brain: more than one elector held isMaster=true at a sampled instant");
        }

        @Override
        public void close() {
            executor.shutdownNow();
        }
    }

    private static int freePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }
}
