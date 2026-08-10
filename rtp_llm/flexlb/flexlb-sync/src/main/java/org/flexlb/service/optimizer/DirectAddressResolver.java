package org.flexlb.service.optimizer;

import java.util.List;

public class DirectAddressResolver implements OptimizerAddressResolver {

    private final List<String> addresses;

    public DirectAddressResolver(String address) {
        this.addresses = List.of(address);
    }

    @Override
    public List<String> getAddresses() {
        return addresses;
    }

    /** Static address is always available; no initialization needed. */
    @Override
    public boolean start() {
        return true;
    }

    @Override
    public void shutdown() {
    }
}
