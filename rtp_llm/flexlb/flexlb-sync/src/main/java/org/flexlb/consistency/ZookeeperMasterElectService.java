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

import static org.flexlb.consistency.LBStatusConsistencyService.MASTER_CHANGE_NOTIFY_PATH;

@Slf4j
@Component
public class ZookeeperMasterElectService implements LeaderSelectorListener {

    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger("syncConsistencyLogger");

    private static final String MASTER_NAMESPACE = "whale-master";
    private static final String MASTER_LEADER_PATH = "/master_lb_leader/";
    @Setter
    private LBConsistencyConfig lbConsistencyConfig;
    private final GeneralHttpNettyService generalHttpNettyService;
    private final EngineHealthReporter engineHealthReporter;
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
    private volatile boolean markOffline;
    private volatile boolean autoRejoin = true;
    private volatile String cachedMasterHostIp;

    private final AtomicReference<CountDownLatch> leaderCloseLatchRef = new AtomicReference<>();

    public ZookeeperMasterElectService(GeneralHttpNettyService generalHttpNettyService,
                                       EngineHealthReporter engineHealthReporter) {

        Logger.warn("Initializing ZookeeperMasterElectService...");

        this.generalHttpNettyService = generalHttpNettyService;
        this.engineHealthReporter = engineHealthReporter;

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
        scheduleMasterUpdateTask();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_INIT);
    }

    private void initializeRoleId() {
        roleId = System.getenv("HIPPO_ROLE");
        if (StringUtils.isBlank(roleId)) {
            throw new RuntimeException("Environment variable HIPPO_ROLE is not set or is blank");
        }
    }

    private void initializeIpAndPort() {
        try {
            localIp = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new RuntimeException("Failed to retrieve local host address", e);
        }
        port = Integer.parseInt(System.getProperty("server.port", "7001"));
    }

    private void initializeLBConsistencyConfig() {
        String configStr = System.getenv("WHALE_SYNC_LB_CONSISTENCY_CONFIG");
        LOGGER.warn("WHALE_SYNC_LB_CONSISTENCY_CONFIG = {}.", configStr);

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
            // 在主节点任务完成后，自动重新参与选举
            leaderSelector.autoRequeue();
        } catch (Exception e) {
            LOGGER.warn("Failed to initialize Zookeeper client and leader selector for roleId: {}, currentHost: {}", roleId,
                    localIp, e);
            closeClient();
            closeLeaderSelector();
            throw new RuntimeException("Initialization failed", e);
        }
    }

    private void scheduleMasterUpdateTask() {
        LBStatusConsistencyService.SCHEDULED_EXECUTOR_SERVICE.scheduleWithFixedDelay(
                this::updateLatestMaster, 0, 5, TimeUnit.SECONDS);
    }

    /**
     * 启动选举过程
     */
    public void start() {
        log.warn("ZKMasterElector roleId:{} currentHost:{} doStart start.", roleId, localIp);
        // 启动主节点选举，向 ZooKeeper 注册并创建临时顺序节点
        leaderSelector.start();
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_START);
        log.warn("ZKMasterElector roleId:{} currentHost:{} doStart finished.", roleId, localIp);
    }

    /**
     * 关闭选举选择器
     */
    public void offline() {
        log.warn("ZKMasterElector roleId:{} currentHost:{} offline start.", roleId, localIp);

        markOffline = true;
        autoRejoin = false;
        reportMasterEvent(ZkMasterEvent.LB_SERVICE_OFFLINE);

        if (!isMaster) {
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
        while (true) {
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
        Logger.warn("ZKMasterElector roleId:{} currentHost:{} cachedMasterHostIp is null.", roleId, localIp);
        return null;
    }

    /**
     * 当前节点已经被选为主节点时的回调方法。
     * 该方法需要阻塞，以保持主节点的身份，直到需要释放主节点的时候，才可以返回
     *
     * @param curatorFramework the client
     */
    @Override
    public void takeLeadership(CuratorFramework curatorFramework) {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} takeLeadership", roleId, localIp);
        if (markOffline) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} markOffline, return.", roleId, localIp);
            return;
        }

        // 成为主节点
        isMaster = true;
        reportMasterEvent(ZkMasterEvent.MASTER_TAKE_LEADERSHIP);

        try {
            CountDownLatch countDownLatch = new CountDownLatch(1);
            leaderCloseLatchRef.set(countDownLatch);
            
            // 主动通知其他参与者当前节点已成为主节点
            activelyNotifyParticipants();

            // 当前线程阻塞，等待主节点关闭后，释放主节点
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
            // 释放领导权
            leaderCloseLatchRef.set(null);
            isMaster = false;
            if (!autoRejoin) {
                closeLeaderSelector();
            }
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} released LeaderShip.", roleId, localIp);
        }
    }

    /**
     * 当连接状态发生变化时的回调方法。
     *
     * @param curatorFramework the client
     * @param connectionState  the new state
     */
    @Override
    public void stateChanged(CuratorFramework curatorFramework, ConnectionState connectionState) {
        LOGGER.warn("ZKMasterElector roleId:{} stateChanged:{}", roleId, connectionState);
        switch (connectionState) {
            case CONNECTED:
                reportMasterEvent(ZkMasterEvent.ZK_CONNECTED);
                break;
            case RECONNECTED:
                reportMasterEvent(ZkMasterEvent.ZK_RECONNECTED);
                break;
            case SUSPENDED:
                reportMasterEvent(ZkMasterEvent.ZK_SUSPENDED);
                throw new CancelLeadershipException();
            case LOST:
                reportMasterEvent(ZkMasterEvent.ZK_LOST);
                clearMasterHost();
                throw new CancelLeadershipException();
            case READ_ONLY:
                reportMasterEvent(ZkMasterEvent.ZK_READ_ONLY);
                break;
        }
    }

    /**
     * 更新获取最新的主节点
     */
    public void updateLatestMaster() {
        synchronized (this) {
            try {
                String leaderId = leaderSelector.getLeader().getId();
                if (StringUtils.isNotBlank(leaderId)) {
                    if (!leaderId.equals(cachedMasterHostIp)) {
                        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} leaderId change from {} to {}.", roleId,
                                localIp,
                                cachedMasterHostIp, leaderId);
                    }
                    cachedMasterHostIp = leaderId;
                }
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} getLeaderID error.", roleId, localIp, e);
            }
        }
    }

    /**
     * 主动通知其他参与者当前节点已成为主节点
     */
    private void activelyNotifyParticipants() {
        try {
            Collection<Participant> participants = leaderSelector.getParticipants();
            for (Participant participant : participants) {
                // 只通知非主节点
                if (!participant.isLeader() && localIp.equals(participant.getId())) {
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

    @Scheduled(fixedRate = 1000)
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

    private void clearMasterHost() {
        synchronized (this) {
            cachedMasterHostIp = null;
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} masterHost cleared due to ZK unavailability.", roleId,
                    localIp
            );
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