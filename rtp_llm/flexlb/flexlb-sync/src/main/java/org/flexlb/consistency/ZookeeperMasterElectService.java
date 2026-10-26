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
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.ZookeeperConsistencyConfig;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.consistency.MasterStatusService.MASTER_CHANGE_NOTIFY_PATH;

@Slf4j
@Component
public class ZookeeperMasterElectService implements LeaderSelectorListener {

    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger("syncConsistencyLogger");

    private static final String MASTER_NAMESPACE = "whale-master";
    private static final String MASTER_LEADER_PATH = "/master_lb_leader/";
    private final ZookeeperConsistencyConfig consistencyConfig;
    private final GeneralHttpNettyService generalHttpNettyService;
    private final EngineHealthReporter engineHealthReporter;
    private final Environment environment;
    private final DeploymentIdentity deploymentIdentity;
    @Setter private String roleId;
    @Setter private String localIp;
    @Setter private int port;

    @Setter(AccessLevel.PACKAGE)
    private CuratorFramework client;

    @Setter(AccessLevel.PACKAGE)
    private LeaderSelector leaderSelector;

    @Getter private volatile boolean isMaster;
    private volatile boolean markOffline;
    private volatile boolean autoRejoin = true;
    private volatile String cachedMasterHostIp;

    private volatile LocalNodeIdentity localNode;

    private final AtomicReference<CountDownLatch> leaderCloseLatchRef = new AtomicReference<>();
    private final Object leadershipGate = new Object();
    private boolean leadershipPermitted = true;
    private long connectionRevision;

    public ZookeeperMasterElectService(
            GeneralHttpNettyService generalHttpNettyService,
            EngineHealthReporter engineHealthReporter,
            Environment environment,
            ConfigService configService,
            DeploymentIdentity deploymentIdentity) {

        Logger.warn("Initializing ZookeeperMasterElectService...");

        this.generalHttpNettyService = generalHttpNettyService;
        this.engineHealthReporter = engineHealthReporter;
        this.environment = environment;
        this.deploymentIdentity = deploymentIdentity;
        this.consistencyConfig =
                configService.loadBalanceConfig().getConsistency()
                                instanceof ZookeeperConsistencyConfig zookeeper
                        ? zookeeper
                        : null;

        init();
    }

    public void init() {
        if (consistencyConfig == null) {
            LOGGER.warn("ZooKeeper consistency is disabled.");
            return;
        }
        initializeRoleId();
        initializeIpAndPort();
        initializeZookeeperClient();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_INIT);
    }

    record LocalNodeIdentity(String hostIp, String serverPort, String roleId) {}

    synchronized LocalNodeIdentity localNodeIdentity() {
        if (localNode != null) {
            return localNode;
        }
        String hostIp;
        try {
            hostIp = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException failure) {
            throw new IllegalStateException("Failed to retrieve local host address", failure);
        }
        String rawPort = environment.getProperty("server.port");
        if (rawPort == null) {
            rawPort = System.getProperty("server.port", "7001");
        }
        localNode =
                new LocalNodeIdentity(
                        hostIp,
                        rawPort,
                        consistencyConfig == null ? null : deploymentIdentity.getDeploymentId());
        return localNode;
    }

    private void initializeRoleId() {
        roleId = localNodeIdentity().roleId();
    }

    private void initializeIpAndPort() {
        var identity = localNodeIdentity();
        localIp = identity.hostIp();
        port = Integer.parseInt(identity.serverPort());
    }

    private void initializeZookeeperClient() {
        try {
            client =
                    CuratorFrameworkFactory.builder()
                            .namespace(MASTER_NAMESPACE)
                            .connectString(consistencyConfig.getConnectString())
                            .sessionTimeoutMs(consistencyConfig.getSessionTimeoutMs())
                            .connectionTimeoutMs(consistencyConfig.getConnectionTimeoutMs())
                            .retryPolicy(new ExponentialBackoffRetry(1000, 3))
                            .build();
            client.start();
            leaderSelector = new LeaderSelector(client, MASTER_LEADER_PATH + roleId, this);
            leaderSelector.setId(localIp);
            // Automatically rejoin election after master task completes
            leaderSelector.autoRequeue();
        } catch (Exception e) {
            LOGGER.warn(
                    "Failed to initialize Zookeeper client and leader selector for roleId: {}, currentHost: {}",
                    roleId,
                    localIp,
                    e);
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
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} single node cluster, skip leadership transfer wait.",
                    roleId,
                    localIp);
            return;
        }

        waitForLeadershipTransfer();
    }

    private boolean isSingleNodeCluster() {
        try {
            return leaderSelector.getParticipants().size() <= 1;
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} error while checking participants, assume single node.",
                    roleId,
                    localIp,
                    e);
            return true;
        }
    }

    @SuppressWarnings("BusyWait")
    private void waitForLeadershipTransfer() {
        LOGGER.warn(
                "ZKMasterElector roleId:{} currentHost:{} waiting for leadership transfer to complete.",
                roleId,
                localIp);

        int waitCount = 0;
        final int MAX_WAIT_COUNT = 30; // 30 seconds max
        while (waitCount < MAX_WAIT_COUNT) {
            try {
                if (!isStillMaster()) {
                    LOGGER.warn(
                            "ZKMasterElector roleId:{} currentHost:{} leadership transferred to {}, waitCount: {}.",
                            roleId,
                            localIp,
                            cachedMasterHostIp,
                            waitCount);
                    return;
                }

                waitCount++;
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} still waiting for leadership transfer, waitCount: {}, currentMaster: {}.",
                        roleId,
                        localIp,
                        waitCount,
                        cachedMasterHostIp);
                Thread.sleep(1000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} wait interrupted, waitCount: {}.",
                        roleId,
                        localIp,
                        waitCount);
                return;
            } catch (Exception e) {
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} error while waiting for leadership transfer, waitCount: {}.",
                        roleId,
                        localIp,
                        waitCount,
                        e);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
        LOGGER.warn(
                "ZKMasterElector roleId:{} currentHost:{} leadership transfer timeout after {} seconds, forcing exit.",
                roleId,
                localIp,
                MAX_WAIT_COUNT);
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
            return localNodeIdentity().hostIp();
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
        synchronized (leadershipGate) {
            if (markOffline || !leadershipPermitted) {
                return;
            }
            leaderCloseLatchRef.set(countDownLatch);
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
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} is interrupted.", roleId, localIp);
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} takeLeadership error.",
                    roleId,
                    localIp,
                    e);
        } finally {
            // Release leadership
            synchronized (leadershipGate) {
                leaderCloseLatchRef.compareAndSet(countDownLatch, null);
                isMaster = false;
            }
            if (!autoRejoin) {
                closeLeaderSelector();
            }
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} released LeaderShip.",
                    roleId,
                    localIp);
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
     * Update and retrieve latest master node
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
            if (!java.util.Objects.equals(previous, next)) {
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} leaderId change from {} to {}.",
                        roleId,
                        localIp,
                        previous,
                        next);
            }
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} getLeaderID error.",
                    roleId,
                    localIp,
                    e);
        }
    }

    private void withdrawLeadership() {
        synchronized (leadershipGate) {
            leadershipPermitted = false;
            connectionRevision++;
            isMaster = false;
            cachedMasterHostIp = null;
            trySignalCloseLatch();
        }
    }

    private void permitLeadership() {
        synchronized (leadershipGate) {
            leadershipPermitted = true;
            connectionRevision++;
        }
    }

    @Scheduled(fixedDelay = 5000L)
    private void updateLatestMasterPeriodically() {
        if (consistencyConfig != null) {
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
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} activelyNotifyParticipants error.",
                    roleId,
                    localIp,
                    e);
        }
    }

    private void notifyParticipant(String participantIp) {
        try {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} notifyParticipant:{}",
                    roleId,
                    localIp,
                    participantIp);
            MasterChangeNotifyReq req = new MasterChangeNotifyReq();
            req.setReqIp(localIp);
            req.setRoleId(roleId);
            URI uri = new URI("http://" + participantIp + ":" + port);
            Mono<MasterChangeNotifyResp> mono =
                    generalHttpNettyService.request(
                            req, uri, MASTER_CHANGE_NOTIFY_PATH, MasterChangeNotifyResp.class);
            mono.timeout(Duration.ofMillis(1000))
                    .toFuture()
                    .whenComplete(
                            (masterChangeNotifyResp, throwable) ->
                                    LOGGER.warn(
                                            "ZKMasterElector roleId:{} currentHost:{} notifyParticipant resp:{}",
                                            roleId,
                                            localIp,
                                            masterChangeNotifyResp,
                                            throwable));
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} notifyParticipant error.",
                    roleId,
                    localIp,
                    e);
        }
    }

    @Scheduled(fixedRate = 2000L)
    private void reportMasterNode() {
        try {
            if (cachedMasterHostIp != null) {
                engineHealthReporter.reportMasterNode(cachedMasterHostIp);
            }
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} reportMasterNode error.",
                    roleId,
                    localIp,
                    e);
        }
    }

    private void reportMasterEvent(ZkMasterEvent event) {
        try {
            engineHealthReporter.reportPrefillBalanceMasterEvent(event);
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} reportMasterEvent error.",
                    roleId,
                    localIp,
                    e);
        }
    }

    private void closeClient() {
        if (client != null) {
            try {
                CloseableUtils.closeQuietly(client);
            } catch (Exception e) {
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} closeClient error.",
                        roleId,
                        localIp,
                        e);
            }
        }
    }

    private void closeLeaderSelector() {
        if (leaderSelector != null) {
            try {
                CloseableUtils.closeQuietly(leaderSelector);
            } catch (Exception e) {
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} closeLeaderSelector error.",
                        roleId,
                        localIp,
                        e);
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
