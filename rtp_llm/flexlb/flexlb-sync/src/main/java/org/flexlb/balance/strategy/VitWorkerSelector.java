package org.flexlb.balance.strategy;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public final class VitWorkerSelector implements VitSelectionStrategy {
    @Override
    public int selectIndex(List<String> addresses) {
        return ThreadLocalRandom.current().nextInt(addresses.size());
    }
}
