package org.flexlb.service.address;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Component;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Collections;
import java.util.Enumeration;

/**
 * Resolves the addresses used to identify a FlexLB instance.
 *
 * <p>The Pod IP is used by ZooKeeper leader election. The instance IP is the
 * routable address returned by DashScope service discovery.
 */
@Slf4j
@Getter
@Component
public class FlexlbInstanceAddressService {

    private static final String POD_IP_ENV = "POD_IP";
    private static final String INSTANCE_NETWORK_INTERFACE = "eth1";

    private final String podIp;
    private final String instanceIp;

    public FlexlbInstanceAddressService() {
        podIp = resolvePodIp();
        instanceIp = resolveInstanceIp(podIp);
        log.info("Resolved FlexLB instance addresses, podIp={}, instanceIp={}", podIp, instanceIp);
    }

    private static String resolvePodIp() {
        String configuredPodIp = System.getenv(POD_IP_ENV);
        if (StringUtils.isNotBlank(configuredPodIp)) {
            return configuredPodIp;
        }
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new IllegalStateException("Failed to resolve FlexLB Pod IP", e);
        }
    }

    private static String resolveInstanceIp(String podIp) {
        try {
            String instanceIp = firstIpv4Address(NetworkInterface.getByName(INSTANCE_NETWORK_INTERFACE));
            if (StringUtils.isNotBlank(instanceIp)) {
                return instanceIp;
            }

            Enumeration<NetworkInterface> networkInterfaces = NetworkInterface.getNetworkInterfaces();
            for (NetworkInterface networkInterface : Collections.list(networkInterfaces)) {
                String candidate = firstIpv4Address(networkInterface);
                if (StringUtils.isNotBlank(candidate) && !candidate.equals(podIp)) {
                    return candidate;
                }
            }
        } catch (SocketException e) {
            log.warn("Failed to resolve FlexLB instance IP; using Pod IP", e);
        }
        return podIp;
    }

    private static String firstIpv4Address(NetworkInterface networkInterface)
            throws SocketException {
        if (networkInterface == null || !networkInterface.isUp()
                || networkInterface.isLoopback()) {
            return null;
        }
        for (InetAddress address : Collections.list(networkInterface.getInetAddresses())) {
            if (address instanceof Inet4Address && !address.isLoopbackAddress()) {
                return address.getHostAddress();
            }
        }
        return null;
    }
}
