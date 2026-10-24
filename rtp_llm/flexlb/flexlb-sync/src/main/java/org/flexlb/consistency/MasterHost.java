package org.flexlb.consistency;

import java.net.InetAddress;
import java.net.Inet4Address;
import java.net.UnknownHostException;
import org.springframework.core.env.Environment;

/**
 * Routable advertised identity shared by ZooKeeper election and follower forwarding.
 * This can differ from the server bind address; wildcard addresses are not identities.
 */
public final class MasterHost {
    private MasterHost() {}

    public static String resolve(Environment environment) {
        String configured = environment.getProperty("flexlb.advertised-ip");
        if (configured == null || configured.isBlank()) {
            configured = environment.getProperty("FLEXLB_ADVERTISED_IP");
        }
        if (configured == null || configured.isBlank()) {
            configured = System.getenv("FLEXLB_ADVERTISED_IP");
        }
        try {
            InetAddress address = configured == null || configured.isBlank()
                    ? InetAddress.getLocalHost() : InetAddress.getByName(configured);
            if (address.isAnyLocalAddress()) {
                throw new IllegalArgumentException("advertised Master address cannot be wildcard");
            }
            if (!(address instanceof Inet4Address)) {
                throw new IllegalArgumentException("advertised Master address requires IPv4");
            }
            return address.getHostAddress();
        } catch (UnknownHostException e) {
            throw new IllegalArgumentException("cannot resolve advertised Master address", e);
        }
    }
}
