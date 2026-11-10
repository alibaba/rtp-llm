package org.flexlb.consistency;

import lombok.Getter;
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
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.consistency.MasterStatusService.MASTER_CHANGE_NOTIFY_PATH;

/**
 * <p>Initialization prepares the ZooKeeper client and leader selector; start joins election.
 * Leadership and connection callbacks update the advertised master. Offline and destroy withdraw
 * participation and close the owned election resources.
 */
@Slf4j
@Component
public class ZookeeperMasterElectService implements LeaderSelectorListener {
    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger("syncConsistencyLogger");
    private static final String MASTER_NAMESPACE = "whale-master";
    private static final String MASTER_LEADER_PATH = "/master_lb_leader/";

    private final ZookeeperConsistencyConfig consistencyConfig;
    private final DeploymentIdentity deploymentIdentity;
    private final GeneralHttpNettyService generalHttpNettyService;
    private final EngineHealthReporter engineHealthReporter;
    private final Environment environment;

    private volatile LocalNodeIdentity localNode;

    private int notificationPort;

    private CuratorFramework client;
    private LeaderSelector leaderSelector;

    @Getter private volatile boolean isMaster;
    private volatile boolean markOffline;

    private volatile String cachedMasterHostIp;

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
                                instanceof ZookeeperConsistencyConfig configured
                        ? configured
                        : null;
        if (!isConsistencyEnabled()) {
            return;
        }
        notificationPort = Integer.parseInt(localNodeIdentity().serverPort());
        initializeZookeeperClient();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_INIT);
    }

    public boolean isConsistencyEnabled() {
        return consistencyConfig != null;
    }

    /**
     * Start election process
     */
    public void start() {
        log.warn(
                "ZKMasterElector roleId:{} currentHost:{} doStart start.", roleId(), localHostIp());
        // Start master election, register with ZooKeeper and create ephemeral sequential node
        leaderSelector.start();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_START);
        log.warn(
                "ZKMasterElector roleId:{} currentHost:{} doStart finished.",
                roleId(),
                localHostIp());
    }

    public String getMasterHostIp() {
        if (isMaster) {
            return localHostIp();
        }
        return cachedMasterHostIp;
    }

    /**
     * Callback method when current node is elected as master.
     * This method must block to maintain master identity until master release is required
     *
     * @param curatorFramework the client
     */
    @Override
    public void takeLeadership(CuratorFramework curatorFramework) {
        LOGGER.warn(
                "ZKMasterElector roleId:{} currentHost:{} takeLeadership", roleId(), localHostIp());
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
            if (!Thread.currentThread().isInterrupted()) {
                countDownLatch.await();
            }
            reportMasterEvent(ZkMasterEvent.MASTER_RELEASE_LEADERSHIP);

        } catch (InterruptedException e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} is interrupted.",
                    roleId(),
                    localHostIp());
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} takeLeadership error.",
                    roleId(),
                    localHostIp(),
                    e);
        } finally {
            // Release leadership
            synchronized (leadershipGate) {
                leaderCloseLatchRef.compareAndSet(countDownLatch, null);
                isMaster = false;
            }
            if (markOffline) {
                closeLeaderSelector();
            }
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} released LeaderShip.",
                    roleId(),
                    localHostIp());
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
        LOGGER.warn("ZKMasterElector roleId:{} stateChanged:{}", roleId(), connectionState);
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
                        roleId(),
                        localHostIp(),
                        previous,
                        next);
            }
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} getLeaderID error.",
                    roleId(),
                    localHostIp(),
                    e);
        }
    }

    public void destroy() {
        LOGGER.warn(
                "ZKMasterElector roleId:{} currentHost:{} destroy start.", roleId(), localHostIp());
        offline();
        closeClient();
        reportMasterEvent(ZkMasterEvent.SERVICE_DESTROY);
        LOGGER.warn(
                "ZKMasterElector roleId:{} currentHost:{} destroy finished.",
                roleId(),
                localHostIp());
    }

    /**
     * Close election selector
     */
    public void offline() {
        log.warn(
                "ZKMasterElector roleId:{} currentHost:{} offline start.", roleId(), localHostIp());

        boolean wasMaster;
        synchronized (leadershipGate) {
            markOffline = true;
            connectionRevision++;
            wasMaster = isMaster;
            isMaster = false;
            cachedMasterHostIp = null;
            trySignalCloseLatch();
        }
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_OFFLINE);

        if (!wasMaster) {
            closeLeaderSelector();
        } else if (isSingleNodeCluster()) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} single node cluster, skip leadership transfer wait.",
                    roleId(),
                    localHostIp());
        } else {
            waitForLeadershipTransfer();
        }

        log.warn(
                "ZKMasterElector roleId:{} currentHost:{} offline finished.",
                roleId(),
                localHostIp());
    }

    synchronized LocalNodeIdentity localNodeIdentity() {
        if (localNode != null) {
            return localNode;
        }
        boolean electionEnabled = isConsistencyEnabled();
        String role = electionEnabled ? deploymentIdentity.getDeploymentId() : null;
        String hostIp;
        try {
            hostIp = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw electionEnabled
                    ? new RuntimeException("Failed to retrieve local host address", e)
                    : new RuntimeException(e);
        }
        String serverPort = environment.getProperty("server.port");
        if (serverPort == null) {
            serverPort = System.getProperty("server.port", "7001");
        }
        localNode = new LocalNodeIdentity(hostIp, serverPort, role);
        return localNode;
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
            leaderSelector = new LeaderSelector(client, MASTER_LEADER_PATH + roleId(), this);
            leaderSelector.setId(localHostIp());
            // Automatically rejoin election after master task completes
            leaderSelector.autoRequeue();
        } catch (Exception e) {
            LOGGER.warn(
                    "Failed to initialize Zookeeper client and leader selector for roleId: {}, currentHost: {}",
                    roleId(),
                    localHostIp(),
                    e);
            closeClient();
            closeLeaderSelector();
            throw new RuntimeException("Initialization failed", e);
        }
    }

    private String roleId() {
        return localNode == null ? null : localNode.roleId();
    }

    private String localHostIp() {
        return localNode == null ? null : localNode.hostIp();
    }

    private void closeClient() {
        if (client != null) {
            try {
                CloseableUtils.closeQuietly(client);
            } catch (Exception e) {
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} closeClient error.",
                        roleId(),
                        localHostIp(),
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
                        roleId(),
                        localHostIp(),
                        e);
            }
        }
    }

    private boolean isSingleNodeCluster() {
        try {
            return leaderSelector.getParticipants().size() <= 1;
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} error while checking participants, assume single node.",
                    roleId(),
                    localHostIp(),
                    e);
            return true;
        }
    }

    @SuppressWarnings("BusyWait")
    private void waitForLeadershipTransfer() {
        LOGGER.warn(
                "ZKMasterElector roleId:{} currentHost:{} waiting for leadership transfer to complete.",
                roleId(),
                localHostIp());

        int waitCount = 0;
        final int MAX_WAIT_COUNT = 30; // 30 seconds max
        while (waitCount < MAX_WAIT_COUNT) {
            try {
                if (!isStillMaster()) {
                    LOGGER.warn(
                            "ZKMasterElector roleId:{} currentHost:{} leadership transferred to {}, waitCount: {}.",
                            roleId(),
                            localHostIp(),
                            cachedMasterHostIp,
                            waitCount);
                    return;
                }

                waitCount++;
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} still waiting for leadership transfer, waitCount: {}, currentMaster: {}.",
                        roleId(),
                        localHostIp(),
                        waitCount,
                        cachedMasterHostIp);
                Thread.sleep(1000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} wait interrupted, waitCount: {}.",
                        roleId(),
                        localHostIp(),
                        waitCount);
                return;
            } catch (Exception e) {
                LOGGER.warn(
                        "ZKMasterElector roleId:{} currentHost:{} error while waiting for leadership transfer, waitCount: {}.",
                        roleId(),
                        localHostIp(),
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
                roleId(),
                localHostIp(),
                MAX_WAIT_COUNT);
    }

    private boolean isStillMaster() {
        updateLatestMaster();
        return localHostIp().equals(cachedMasterHostIp);
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

    private void trySignalCloseLatch() {
        CountDownLatch latch = leaderCloseLatchRef.get();
        if (latch != null) {
            latch.countDown();
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
        if (isConsistencyEnabled()) {
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
                if (!participant.isLeader() && !localHostIp().equals(participant.getId())) {
                    notifyParticipant(participant.getId());
                }
            }
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} activelyNotifyParticipants error.",
                    roleId(),
                    localHostIp(),
                    e);
        }
    }

    private void notifyParticipant(String participantIp) {
        try {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} notifyParticipant:{}",
                    roleId(),
                    localHostIp(),
                    participantIp);
            MasterChangeNotifyReq req = new MasterChangeNotifyReq();
            req.setReqIp(localHostIp());
            req.setRoleId(roleId());
            URI uri = new URI("http://" + participantIp + ":" + notificationPort);
            Mono<MasterChangeNotifyResp> mono =
                    generalHttpNettyService.request(
                            req, uri, MASTER_CHANGE_NOTIFY_PATH, MasterChangeNotifyResp.class);
            mono.timeout(Duration.ofMillis(1000))
                    .toFuture()
                    .whenComplete(
                            (masterChangeNotifyResp, throwable) ->
                                    LOGGER.warn(
                                            "ZKMasterElector roleId:{} currentHost:{} notifyParticipant resp:{}",
                                            roleId(),
                                            localHostIp(),
                                            masterChangeNotifyResp,
                                            throwable));
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} notifyParticipant error.",
                    roleId(),
                    localHostIp(),
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
                    roleId(),
                    localHostIp(),
                    e);
        }
    }

    private void reportMasterEvent(ZkMasterEvent event) {
        try {
            engineHealthReporter.reportPrefillBalanceMasterEvent(event);
        } catch (Exception e) {
            LOGGER.warn(
                    "ZKMasterElector roleId:{} currentHost:{} reportMasterEvent error.",
                    roleId(),
                    localHostIp(),
                    e);
        }
    }

    record LocalNodeIdentity(String hostIp, String serverPort, String roleId) {}
}
