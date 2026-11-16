package org.flexlb.consistency;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.leader.CancelLeadershipException;
import org.apache.curator.framework.recipes.leader.LeaderSelector;
import org.apache.curator.framework.recipes.leader.LeaderSelectorListener;
import org.apache.curator.framework.recipes.leader.Participant;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.apache.curator.utils.CloseableUtils;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.Collection;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.consistency.LBStatusConsistencyService.MASTER_CHANGE_NOTIFY_PATH;

/**
 * ZooKeeper-backed master election per role: wins the leader lock, blocks in
 * {@link #takeLeadership} while master duty runs, and answers master identity
 * from the cached leader lookup. Offline or a suspended ZK session withdraws
 * leadership and fences stale lookups by connection revision, so observers
 * never schedule against a revoked connection.
 */
@Slf4j
@Component
public class ZookeeperMasterElectService implements LeaderSelectorListener {

    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger("syncConsistencyLogger");

    private static final String MASTER_NAMESPACE = "whale-master";
    private static final String MASTER_LEADER_PATH = "/master_lb_leader/";
    @Getter
    @Setter
    private LBConsistencyConfig lbConsistencyConfig;
    private final GeneralHttpNettyService generalHttpNettyService;
    private final EngineHealthReporter engineHealthReporter;
    private final Environment environment;
    @Setter
    private String roleId;
    @Setter
    private String localIp;
    @Setter
    private int port;
    @Setter(AccessLevel.PACKAGE)
    private CuratorFramework client;
    @Setter(AccessLevel.PACKAGE)
    private LeaderSelector leaderSelector;
    @Getter
    private volatile boolean isMaster;
    @Getter
    private volatile long leadershipEpoch;
    private volatile boolean markOffline;
    private volatile boolean autoRejoin = true;
    private volatile String cachedMasterHostIp;

    private final AtomicReference<CountDownLatch> leaderCloseLatchRef = new AtomicReference<>();

    /**
     * Serializes leadership acceptance/release and leader-lookup publication.
     * {@code connectionRevision} fences pre-disconnect lookups: a lookup that
     * started before a session suspension must not publish the old leader
     * after the connection was replaced.
     */
    private final Object leadershipGate = new Object();
    private boolean leadershipPermitted = true;
    private long connectionRevision;

    public ZookeeperMasterElectService(GeneralHttpNettyService generalHttpNettyService,
                                       EngineHealthReporter engineHealthReporter,
                                       Environment environment) {

        Logger.warn("Initializing ZookeeperMasterElectService...");

        this.generalHttpNettyService = generalHttpNettyService;
        this.engineHealthReporter = engineHealthReporter;
        this.environment = environment;

        init();
    }

    public void init() {
        initializeLBConsistencyConfig();
        if (!lbConsistencyConfig.isNeedConsistency()) {
            LOGGER.warn("Consistency is not required for LBConsistencyConfig.");
            return;
        }
        initializeRoleId();
        initializeIpAndPort();
        initializeZookeeperClient();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_INIT);
    }

    private void initializeRoleId() {
        roleId = System.getenv("HIPPO_ROLE");
        if (StringUtils.isBlank(roleId)) {
            throw new IllegalArgumentException("HIPPO_ROLE is required when needConsistency=true");
        }
    }

    private void initializeIpAndPort() {
        try {
            localIp = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new RuntimeException("Failed to retrieve local host address", e);
        }
        // Read from Spring Environment to respect --server.port= CLI args;
        // fall back to JVM system property.
        String portStr = environment.getProperty("server.port");
        if (portStr == null) {
            portStr = System.getProperty("server.port", "7001");
        }
        port = Integer.parseInt(portStr);
    }

    private void initializeLBConsistencyConfig() {
        String configStr = System.getenv("FLEXLB_SYNC_CONSISTENCY_CONFIG");
        LOGGER.warn("FLEXLB_SYNC_CONSISTENCY_CONFIG = {}.", configStr);

        lbConsistencyConfig = configStr == null
                ? new LBConsistencyConfig()
                : JsonUtils.toObject(configStr, LBConsistencyConfig.class);
    }

    private void initializeZookeeperClient() {
        try {
            LBConsistencyConfig.ZookeeperConfig zookeeperConfig = lbConsistencyConfig.getZookeeperConfig();
            client = CuratorFrameworkFactory.builder()
                    .namespace(MASTER_NAMESPACE)
                    .connectString(zookeeperConfig.getZkHost())
                    .sessionTimeoutMs(zookeeperConfig.getZkTimeoutMs())
                    .connectionTimeoutMs(zookeeperConfig.getZkTimeoutMs())
                    .retryPolicy(new ExponentialBackoffRetry(1000, 3))
                    .build();
            client.start();
            leaderSelector = new LeaderSelector(client, MASTER_LEADER_PATH + roleId, this);
            leaderSelector.setId(localIp);
            // Automatically rejoin election after master task completes
            leaderSelector.autoRequeue();
        } catch (Exception e) {
            LOGGER.warn("Failed to initialize Zookeeper client and leader selector for roleId: {}, currentHost: {}", roleId,
                    localIp, e);
            closeClient();
            closeLeaderSelector();
            throw new RuntimeException("Initialization failed", e);
        }
    }

    /**
     * Start election process
     */
    public void start() {
        log.warn("ZKMasterElector roleId:{} currentHost:{} doStart start.", roleId, localIp);
        // Start master election, register with ZooKeeper and create ephemeral sequential node
        leaderSelector.start();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_START);
        log.warn("ZKMasterElector roleId:{} currentHost:{} doStart finished.", roleId, localIp);
    }

    /**
     * Close election selector
     */
    public void offline() {
        log.warn("ZKMasterElector roleId:{} currentHost:{} offline start.", roleId, localIp);

        // Withdraw authority under the same gate that accepts leadership, so
        // a concurrent takeLeadership can never observe a half-released state.
        boolean wasMaster;
        synchronized (leadershipGate) {
            markOffline = true;
            autoRejoin = false;
            connectionRevision++;
            wasMaster = isMaster;
            isMaster = false;
            cachedMasterHostIp = null;
            trySignalCloseLatch();
        }
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_OFFLINE);

        if (!wasMaster) {
            closeLeaderSelector();
        } else {
            handleMasterOffline();
        }

        log.warn("ZKMasterElector roleId:{} currentHost:{} offline finished.", roleId, localIp);
    }

    private void handleMasterOffline() {
        trySignalCloseLatch();

        if (isSingleNodeCluster()) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} single node cluster, skip leadership transfer wait.",
                    roleId, localIp);
            return;
        }

        waitForLeadershipTransfer();
    }

    private boolean isSingleNodeCluster() {
        try {
            return leaderSelector.getParticipants().size() <= 1;
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} error while checking participants, assume single node.",
                    roleId, localIp, e);
            return true;
        }
    }

    @SuppressWarnings("BusyWait")
    private void waitForLeadershipTransfer() {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} waiting for leadership transfer to complete.", roleId, localIp);

        int waitCount = 0;
        final int MAX_WAIT_COUNT = 30;  // 30 seconds max
        while (waitCount < MAX_WAIT_COUNT) {
            try {
                if (!isStillMaster()) {
                    LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} leadership transferred to {}, waitCount: {}.",
                            roleId, localIp, cachedMasterHostIp, waitCount);
                    return;
                }

                waitCount++;
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} still waiting for leadership transfer, waitCount: {}, currentMaster: {}.",
                        roleId, localIp, waitCount, cachedMasterHostIp);
                Thread.sleep(1000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} wait interrupted, waitCount: {}.", roleId, localIp, waitCount);
                return;
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} error while waiting for leadership transfer, waitCount: {}.",
                        roleId, localIp, waitCount, e);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} leadership transfer timeout after {} seconds, forcing exit.",
                roleId, localIp, MAX_WAIT_COUNT);
    }

    private boolean isStillMaster() {
        updateLatestMaster();
        return localIp.equals(cachedMasterHostIp);
    }

    public String getMasterHostIp(boolean forceSync) {
        if (forceSync) {
            updateLatestMaster();
        }
        if (isMaster) {
            return localIp;
        }
        if (cachedMasterHostIp != null) {
            return cachedMasterHostIp;
        }
        return null;
    }

    /**
     * Callback method when current node is elected as master.
     * This method must block to maintain master identity until master release is required
     *
     * @param curatorFramework the client
     */
    @Override
    public void takeLeadership(CuratorFramework curatorFramework) {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} takeLeadership", roleId, localIp);
        CountDownLatch countDownLatch = new CountDownLatch(1);
        // Publish the stop signal and accept leadership atomically: a session
        // suspended after this gate cannot leave the latch unpublished while
        // isMaster is already visible to other threads.
        synchronized (leadershipGate) {
            if (markOffline || !leadershipPermitted) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} leadership withdrawn, return.", roleId, localIp);
                return;
            }
            leaderCloseLatchRef.set(countDownLatch);
            leadershipEpoch++;
            isMaster = true;
        }

        try {
            reportMasterEvent(ZkMasterEvent.MASTER_TAKE_LEADERSHIP);
            // Actively notify other participants that current node has become master
            activelyNotifyParticipants();

            // Current thread blocks, waiting for master shutdown before releasing master
            while (!Thread.currentThread().isInterrupted()) {
                if (countDownLatch.await(1000, TimeUnit.MILLISECONDS)) {
                    break;
                }
            }
            reportMasterEvent(ZkMasterEvent.MASTER_RELEASE_LEADERSHIP);

        } catch (InterruptedException e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} is interrupted.", roleId, localIp);
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} takeLeadership error.", roleId, localIp, e);
        } finally {
            // Release leadership
            synchronized (leadershipGate) {
                leaderCloseLatchRef.compareAndSet(countDownLatch, null);
                isMaster = false;
            }
            if (!autoRejoin) {
                closeLeaderSelector();
            }
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} released LeaderShip.", roleId, localIp);
        }
    }

    /**
     * Callback method when connection state changes.
     *
     * @param curatorFramework the client
     * @param connectionState  the new state
     */
    @Override
    public void stateChanged(CuratorFramework curatorFramework, ConnectionState connectionState) {
        LOGGER.warn("ZKMasterElector roleId:{} stateChanged:{}", roleId, connectionState);
        switch (connectionState) {
            case CONNECTED:
                permitLeadership();
                reportMasterEvent(ZkMasterEvent.ZK_CONNECTED);
                break;
            case RECONNECTED:
                permitLeadership();
                reportMasterEvent(ZkMasterEvent.ZK_RECONNECTED);
                break;
            case SUSPENDED:
                withdrawLeadership();
                reportMasterEvent(ZkMasterEvent.ZK_SUSPENDED);
                throw new CancelLeadershipException();
            case LOST:
                withdrawLeadership();
                reportMasterEvent(ZkMasterEvent.ZK_LOST);
                throw new CancelLeadershipException();
            case READ_ONLY:
                reportMasterEvent(ZkMasterEvent.ZK_READ_ONLY);
                break;
        }
    }

    /**
     * Update and retrieve latest master node. The method-level monitor
     * serializes concurrent lookups; the revision fence drops a lookup that
     * started before a session suspension or offline, so the pre-disconnect
     * leader cannot be re-published over a newer connection's answer.
     */
    public synchronized void updateLatestMaster() {
        long revision;
        synchronized (leadershipGate) {
            if (!leadershipPermitted) {
                return;
            }
            revision = connectionRevision;
        }
        try {
            String leaderId = leaderSelector.getLeader().getId();
            String next = StringUtils.isBlank(leaderId) ? null : leaderId;
            String previous;
            synchronized (leadershipGate) {
                if (!leadershipPermitted || revision != connectionRevision) {
                    return;
                }
                previous = cachedMasterHostIp;
                cachedMasterHostIp = next;
            }
            if (!Objects.equals(previous, next)) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} leaderId change from {} to {}.", roleId,
                        localIp, previous, next);
            }
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} getLeaderID error.", roleId, localIp, e);
        }
    }

    /**
     * Revoke master authority for the current connection: withdraw
     * {@code isMaster}, forget the cached leader, and signal the master
     * callback to exit. Called before {@code CancelLeadershipException} is
     * thrown from {@link #stateChanged} so disconnected-session observers
     * never run with authority.
     */
    private void withdrawLeadership() {
        synchronized (leadershipGate) {
            leadershipPermitted = false;
            connectionRevision++;
            isMaster = false;
            cachedMasterHostIp = null;
            trySignalCloseLatch();
        }
    }

    /** Admit leadership candidates again after a (re)connected session. */
    private void permitLeadership() {
        synchronized (leadershipGate) {
            leadershipPermitted = true;
            connectionRevision++;
        }
    }

    @Scheduled(fixedDelay = 5000L)
    private void updateLatestMasterPeriodically() {
        if (lbConsistencyConfig.isNeedConsistency()) {
            updateLatestMaster();
        }
    }

    /**
     * Actively notify other participants that current node has become master
     */
    private void activelyNotifyParticipants() {
        try {
            Collection<Participant> participants = leaderSelector.getParticipants();
            for (Participant participant : participants) {
                // Only notify non-master participants
                if (!participant.isLeader() && !localIp.equals(participant.getId())) {
                    notifyParticipant(participant.getId());
                }
            }
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} activelyNotifyParticipants error.", roleId, localIp, e);
        }
    }

    private void notifyParticipant(String participantIp) {
        try {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} notifyParticipant:{}", roleId, localIp, participantIp);
            MasterChangeNotifyReq req = new MasterChangeNotifyReq();
            req.setReqIp(localIp);
            req.setRoleId(roleId);
            URI uri = new URI("http://" + participantIp + ":" + port);
            Mono<MasterChangeNotifyResp> mono =
                    generalHttpNettyService.request(req, uri, MASTER_CHANGE_NOTIFY_PATH, MasterChangeNotifyResp.class);
            mono.timeout(Duration.ofMillis(1000))
                    .toFuture()
                    .whenComplete((masterChangeNotifyResp, throwable) ->
                            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} notifyParticipant resp:{}", roleId,
                                    localIp,
                                    masterChangeNotifyResp, throwable));
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} notifyParticipant error.", roleId, localIp, e);
        }
    }

    @Scheduled(fixedRate = 2000L)
    private void reportMasterNode() {
        try {
            if (cachedMasterHostIp != null) {
                engineHealthReporter.reportMasterNode(cachedMasterHostIp);
            }
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} reportMasterNode error.", roleId, localIp, e);
        }
    }

    private void reportMasterEvent(ZkMasterEvent event) {
        try {
            engineHealthReporter.reportPrefillBalanceMasterEvent(event);
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} reportMasterEvent error.", roleId, localIp, e);
        }
    }

    private void closeClient() {
        if (client != null) {
            try {
                CloseableUtils.closeQuietly(client);
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} closeClient error.", roleId, localIp, e);
            }
        }
    }

    private void closeLeaderSelector() {
        if (leaderSelector != null) {
            try {
                CloseableUtils.closeQuietly(leaderSelector);
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} closeLeaderSelector error.", roleId, localIp, e);
            }
        }
    }

    private void trySignalCloseLatch() {
        CountDownLatch latch = leaderCloseLatchRef.get();
        if (latch != null) {
            latch.countDown();
        }
    }

    public void destroy() {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} destroy start.", roleId, localIp);
        offline();
        closeClient();
        reportMasterEvent(ZkMasterEvent.SERVICE_DESTROY);
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} destroy finished.", roleId, localIp);
    }
}
