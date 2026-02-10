package org.flexlb.consistency;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class LBStatusConsistencyService implements MasterElectService {

    public static final String MASTER_CHANGE_NOTIFY_PATH = "/rtp_llm/notify_master";
    public static final ScheduledExecutorService SCHEDULED_EXECUTOR_SERVICE = new ScheduledThreadPoolExecutor(
            4,
            new NamedThreadFactory("LBStatusConsistencyService-Schedule-Thread"),
            new ThreadPoolExecutor.AbortPolicy()
    );

    private final ZookeeperMasterElectService zookeeperMasterElectService;
    private LBConsistencyConfig lbConsistencyConfig;
    private String serverPort;
    private String roleId;

    public LBStatusConsistencyService(ZookeeperMasterElectService zookeeperMasterElectService) {
        this.zookeeperMasterElectService = zookeeperMasterElectService;
        this.init();
    }

    public void init() {
        log.info("start init LBStatusConsistencyService.");
        String hostIp;
        try {
            hostIp = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new RuntimeException(e);
        }
        serverPort = System.getProperty("server.port", "7001");
        log.info("hostIp:{}, serverPort:{}.", hostIp, serverPort);
        roleId = System.getenv("HIPPO_ROLE");
        if (StringUtils.isBlank(roleId)) {
            throw new RuntimeException("HIPPO_ROLE env is blank");
        }
        String configStr = System.getenv("WHALE_SYNC_LB_CONSISTENCY_CONFIG");
        log.info("WHALE_SYNC_LB_CONSISTENCY_CONFIG = {}.", configStr);
        if (configStr == null) {
            lbConsistencyConfig = new LBConsistencyConfig();
        } else {
            lbConsistencyConfig = JsonUtils.toObject(configStr, LBConsistencyConfig.class);
        }
        if (!isNeedConsistency()) {
            log.warn("LBStatusConsistencyService is not need.");
            return;
        }
        log.info("start init ZookeeperMasterElectService.");

        SCHEDULED_EXECUTOR_SERVICE.scheduleWithFixedDelay(this::syncLBStatusFromMaster, 1000, 500, TimeUnit.MILLISECONDS);
    }

    @Override
    public void start() {
        if (!isNeedConsistency()) {
            log.warn("start: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.start();
    }

    @Override
    public void offline() {
        if (!isNeedConsistency()) {
            log.warn("offline: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.offline();
    }

    @Override
    public void destroy() {
        if (!isNeedConsistency()) {
            log.warn("destroy: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.destroy();
    }

    @Override
    public boolean isNeedConsistency() {
        return lbConsistencyConfig.isNeedConsistency();
    }

    @Override
    public boolean isMaster() {
        if (!isNeedConsistency()) {
            return false;
        }
        return zookeeperMasterElectService.isMaster();
    }

    @Override
    public void refreshMasterHost(boolean forceSync) {
        if (isNeedConsistency() && forceSync) {
            zookeeperMasterElectService.updateLatestMaster();
        }
    }

    public String getMasterHostIpPort() {
        if (!isNeedConsistency()) {
            return null;
        }
        String masterHostIp = zookeeperMasterElectService.getMasterHostIp(false);
        if (masterHostIp == null) {
            Logger.warn("getMasterHostIpPort: masterHostIp is null.");
            return null;
        }
        return masterHostIp + ":" + serverPort;
    }

    /**
     * Handle master change
     *
     * @param req MasterChangeNotifyReq
     * @return MasterChangeNotifyResp
     */
    public MasterChangeNotifyResp handleMasterChange(MasterChangeNotifyReq req) {
        log.warn("recv MasterChangeNotifyReq:{}.", req);
        if (!roleId.equals(req.getRoleId())) {
            MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
            resp.setSuccess(false);
            resp.setMsg("roleId not match this:" + roleId);
            return resp;
        }
        this.refreshMasterHost(true);
        MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
        resp.setSuccess(true);
        return resp;
    }

    public SyncLBStatusResp dumpLBStatus() {
        SyncLBStatusResp resp = new SyncLBStatusResp();
        resp.setSuccess(true);
        // TODO Get master status
        return resp;
    }

    /**
     * Slave node syncs LB status from master node
     */
    private void syncLBStatusFromMaster() {
        // TODO Get master status
    }
}
