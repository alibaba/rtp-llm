package org.flexlb.consistency;

/**
 * @author zjw
 * description:
 * date: 2025/3/20
 */
public interface MasterElectService {

    void start();

    void offline();

    void destroy();

    boolean isMaster();

    String getMasterHostIp(boolean forceSync);

}
