package org.flexlb.service.optimizer;

import java.util.List;

public interface OptimizerAddressResolver {

    List<String> getAddresses();

    void shutdown();
}
