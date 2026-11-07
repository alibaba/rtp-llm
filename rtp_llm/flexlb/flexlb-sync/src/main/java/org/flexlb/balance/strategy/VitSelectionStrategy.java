package org.flexlb.balance.strategy;

import java.util.List;

@FunctionalInterface
public interface VitSelectionStrategy {
    /** Choose an index from the nonempty, read-only address list; do not retain the list. */
    int selectIndex(List<String> addresses);
}
