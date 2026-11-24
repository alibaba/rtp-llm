package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PendingPrefillPrefixesTest {
    private final List<GlobalQueueEntry> ready = new ArrayList<>();
    private final PendingPrefillPrefixes prefixes = new PendingPrefillPrefixes(ready::add);
    private final ServerStatus worker = new ServerStatus();
    private long sequence;

    PendingPrefillPrefixesTest() {
        worker.setRole(RoleType.PREFILL);
        worker.setGroup("g0");
    }

    @Test
    void aLowerPriorityExactMatchDoesNotHideAnEligibleProducer() {
        publish(10, 1L, 2L, 3L);
        var follower = entry(50, 1L, 2L, 3L);
        assertFalse(prefixes.awaitBetterPrefix(follower, worker, 0, 0));
        var producer = publish(60, 1L, 2L, 4L);
        assertTrue(prefixes.awaitBetterPrefix(follower, worker, 1, 0));
        assertTrue(prefixes.finished(producer, true));
        assertEquals(List.of(follower), ready);
        assertFalse(prefixes.isWaiting(follower));
    }

    @Test
    void budgetExpiryWakesTheFollowerExactlyOnceEvenIfTheProducerFinishesLater() {
        var producer = publish(50, 1L, 2L);
        var follower = entry(50, 1L, 2L);
        assertTrue(prefixes.awaitBetterPrefix(follower, worker, 0, 0));
        assertEquals(Long.MAX_VALUE, prefixes.resumeExpired(60_000));
        assertFalse(prefixes.finished(producer, true));
        assertEquals(List.of(follower), ready);
    }

    private RequestExecution publish(int priority, Long... keys) {
        RequestExecution execution = mock(RequestExecution.class);
        when(execution.prefill()).thenReturn(worker);
        prefixes.published(entry(priority, keys), execution, 0);
        return execution;
    }

    private GlobalQueueEntry entry(int priority, Long... keys) {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(60_000);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        var ordering = QueueOrderingConfig.priority();
        ordering.setPrefillCostOrder(true);
        config.queueScheduler().setOrdering(ordering);
        var context = new BalanceContext(config);
        var request = new Request();
        request.setSeqLen(keys.length);
        request.setCacheKeyBlockSize(1);
        request.setBlockCacheKeys(List.of(keys));
        context.setRequest(request);
        var entry = new GlobalQueueEntry(context, new CompletableFuture<>(), priority);
        entry.sequence = ++sequence;
        return entry;
    }
}
