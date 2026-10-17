package org.flexlb.consistency;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.LinkedHashMap;
import java.util.Map;

/** Local node and elected master status exposed to routing and administration. */
@Slf4j
@Component
public class MasterStatusService implements MasterStatusView {

    public static final String MASTER_CHANGE_NOTIFY_PATH = "/rtp_llm/notify_master";

    private final ZookeeperMasterElectService zookeeperMasterElectService;
    private final LBConsistencyConfig lbConsistencyConfig;
    private final ZookeeperMasterElectService.LocalNodeIdentity localNode;

    public MasterStatusService(ZookeeperMasterElectService zookeeperMasterElectService) {
        this.zookeeperMasterElectService = zookeeperMasterElectService;
        lbConsistencyConfig = zookeeperMasterElectService.getLbConsistencyConfig();
        localNode = zookeeperMasterElectService.localNodeIdentity();
    }

    public void start() {
        if (!isNeedConsistency()) {
            log.warn("start: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.start();
    }

    public void offline() {
        if (!isNeedConsistency()) {
            log.warn("offline: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.offline();
    }

    @PreDestroy
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

    public String getMasterHostIpPort() {
        if (!isNeedConsistency()) {
            return null;
        }
        String masterHostIp = zookeeperMasterElectService.getMasterHostIp();
        if (masterHostIp == null) {
            return null;
        }
        return masterHostIp + ":" + localNode.serverPort();
    }

    public String getLocalHostIp() {
        return localNode.hostIp();
    }

    /**
     * Handle master change
     *
     * @param req MasterChangeNotifyReq
     * @return MasterChangeNotifyResp
     */
    public MasterChangeNotifyResp handleMasterChange(MasterChangeNotifyReq req) {
        log.warn("recv MasterChangeNotifyReq:{}.", req);
        if (!isNeedConsistency() || !localNode.roleId().equals(req.getRoleId())) {
            MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
            resp.setSuccess(false);
            resp.setMsg("roleId not match this:" + localNode.roleId());
            return resp;
        }
        zookeeperMasterElectService.updateLatestMaster();
        MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
        resp.setSuccess(true);
        return resp;
    }

    public SyncLBStatusResp dumpLBStatus() {
        SyncLBStatusResp resp = new SyncLBStatusResp();
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("consistency_enabled", isNeedConsistency());
        snapshot.put("master", isMaster());
        snapshot.put("local_host", localNode.hostIp());
        snapshot.put("master_host", getMasterHostIpPort());
        snapshot.put("server_port", localNode.serverPort());
        resp.setSuccess(true);
        resp.setMsg("leadership snapshot");
        resp.setLbStatus(JsonUtils.toStringOrEmpty(snapshot));
        return resp;
    }
}
