package org.flexlb.constraint;

import org.flexlb.constraint.source.SidBucketClient;
import org.junit.jupiter.api.Test;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class IgraphConstraintTreePollerTest {
    @Test
    void explicitEmptyPrefixNumericConfigAndDryRunNeverPublish() {
        var builds = builds();
        var leader = mock(org.flexlb.consistency.LBStatusConsistencyService.class);
        when(leader.isMaster()).thenReturn(true);
        var env = new org.springframework.mock.env.MockEnvironment()
                .withProperty("constraint.tree.igraph.model", "engine_service")
                .withProperty("constraint.tree.igraph.key.prefix", "")
                .withProperty("constraint.tree.igraph.bucket.algorithm", "ITEM_ID_MOD")
                .withProperty("constraint.tree.igraph.bucket.count", "4000")
                .withProperty("constraint.tree.igraph.source.row.limit", "2000")
                .withProperty("constraint.tree.igraph.dry.run", "true");
        var calls = new AtomicInteger();
        SidBucketClient client = (k, l, t) -> {
            calls.incrementAndGet();
            return CompletableFuture.completedFuture(List.of(new SidBucketClient.Row(k, k, "C1C2")));
        };
        var poller = new IgraphConstraintTreePoller(client, builds, leader, env);
        try {
            poller.pollOnce();
            assertEquals("VALIDATED_NO_PUBLISH", poller.getStatus().state());
            assertEquals(4000, calls.get());
            assertEquals(4000, poller.getStatus().items());
            assertEquals(0, poller.getStatus().submittedVersion());
            assertTrue(poller.getStatus().message().contains("maxBucketRows=1"));
            verifyNoInteractions(builds);
            when(leader.isMaster()).thenReturn(false);
            poller.pollOnce();
            assertEquals(4000, calls.get());
        } finally { poller.close(); }
        var missingPrefix = new org.springframework.mock.env.MockEnvironment()
                .withProperty("constraint.tree.igraph.model", "engine_service");
        assertThrows(IllegalArgumentException.class,
                () -> new IgraphConstraintTreePoller(client, builds, leader, missingPrefix));
        env.withProperty("constraint.tree.igraph.key.prefix", "").withProperty("constraint.tree.igraph.bucket.algorithm", "TYPO");
        assertThrows(IllegalArgumentException.class, () -> new IgraphConstraintTreePoller(client, builds, leader, env));
    }

    @Test
    void serverCapFailureRetainsTreeEvenWithPublicationEnabled() {
        var builds = builds();
        var reader = new BucketSidReader((k, l, t) -> CompletableFuture.completedFuture(
                List.of(new SidBucketClient.Row(k, k, "C1C2"))), BucketSidReaderTest.numericSettings(4000, 1));
        var poller = new IgraphConstraintTreePoller(reader, builds, () -> true, "gul_item", true, true, 600, CLOCK);
        try {
            poller.pollOnce();
            assertEquals("FAILED", poller.getStatus().state());
            assertTrue(poller.getStatus().message().contains("possible truncation"));
            verify(builds, never()).submit(any());
        } finally { poller.close(); }
    }

    private static final Clock CLOCK = Clock.fixed(Instant.ofEpochMilli(100), ZoneOffset.UTC);

    private ConstraintTreeBuildService builds() {
        var service = mock(ConstraintTreeBuildService.class);
        when(service.getStatus()).thenReturn(new ConstraintTreeModels.BuildStatus(
                ConstraintTreeModels.BuildState.READY, 150, 150, 149, 1, 3, 1, 1, "ready"));
        when(service.submit(any())).thenAnswer(call -> {
            ConstraintTreeModels.BuildRequest request = call.getArgument(0);
            return new ConstraintTreeModels.Submission(ConstraintTreeModels.SubmissionState.ACCEPTED,
                    request.version(), request.version(), "accepted");
        });
        return service;
    }

    private BucketSidReader reader(SidBucketClient client) {
        return new BucketSidReader(client, BucketSidReaderTest.settings(1, 1, 10, 0));
    }

    @Test
    void completeReadUsesExistingBuildPipelineAndMonotonicVersion() {
        var builds = builds();
        var reader = reader((k, l, t) -> CompletableFuture.completedFuture(
                List.of(new SidBucketClient.Row(k, "1", "C1C2"))));
        var poller = new IgraphConstraintTreePoller(reader, builds, () -> true, "gul_item", true, true, 600, CLOCK);
        try {
            poller.pollOnce();
            assertEquals("SUBMITTED", poller.getStatus().state());
            assertEquals(151, poller.getStatus().submittedVersion());
            verify(builds).submit(argThat(r -> r.version() == 151 && r.model().equals("gul_item")
                    && r.sids().equals(List.of("C1C2")) && r.rqTokenIds() == null));
        } finally { poller.close(); }
    }

    @Test
    void failedReadNeverSubmitsAndLeavesTreeUntouched() {
        var builds = builds();
        var reader = reader((k, l, t) -> CompletableFuture.failedFuture(new IllegalStateException("unavailable")));
        var poller = new IgraphConstraintTreePoller(reader, builds, () -> true, "gul_item", true, true, 600, CLOCK);
        try {
            poller.pollOnce();
            assertEquals("FAILED", poller.getStatus().state());
            verify(builds, never()).submit(any());
        } finally { poller.close(); }
    }

    @Test
    void noReadsBeforeInitializationAcknowledgementOrOnFollower() {
        var calls = new AtomicInteger();
        var reader = reader((k, l, t) -> { calls.incrementAndGet(); return CompletableFuture.completedFuture(List.of()); });
        for (boolean[] flags : List.of(new boolean[]{false, true, true},
                new boolean[]{true, false, true}, new boolean[]{true, true, false})) {
            var builds = builds();
            var poller = new IgraphConstraintTreePoller(reader, builds, () -> flags[0],
                    "gul_item", flags[1], flags[2], 600, CLOCK);
            try {
                poller.pollOnce();
                assertFalse(poller.trigger());
                verify(builds, never()).submit(any());
            } finally { poller.close(); }
        }
        assertEquals(0, calls.get());
    }

    @Test
    void repeatedTriggersDoNotOverlapAndCloseCancelsRead() throws Exception {
        var future = new CompletableFuture<List<SidBucketClient.Row>>();
        var entered = new java.util.concurrent.CountDownLatch(1);
        var poller = new IgraphConstraintTreePoller(reader((k, l, t) -> { entered.countDown(); return future; }),
                builds(), () -> true, "gul_item", true, true, 600, CLOCK);
        try {
            assertTrue(poller.trigger());
            assertTrue(entered.await(2, java.util.concurrent.TimeUnit.SECONDS));
            assertFalse(poller.trigger());
        } finally { poller.close(); }
        assertFalse(poller.trigger());
    }

    @Test
    void automaticPollingSurvivesTransientLeadershipLookupFailure() throws Exception {
        var queries = new AtomicInteger();
        var checks = new AtomicInteger();
        var builds = builds();
        var submitted = new java.util.concurrent.CountDownLatch(1);
        doAnswer(call -> {
            ConstraintTreeModels.BuildRequest request = call.getArgument(0);
            submitted.countDown();
            return new ConstraintTreeModels.Submission(ConstraintTreeModels.SubmissionState.ACCEPTED,
                    request.version(), request.version(), "accepted");
        }).when(builds).submit(any());
        var source = reader((k, l, t) -> {
            queries.incrementAndGet();
            return CompletableFuture.completedFuture(List.of(new SidBucketClient.Row(k, "1", "C1C2")));
        });
        var poller = new IgraphConstraintTreePoller(source, builds, () -> {
            if (checks.getAndIncrement() == 0) { throw new IllegalStateException("leader temporarily unavailable"); }
            return true;
        }, "gul_item", true, true, 1, CLOCK);
        try {
            poller.start();
            assertTrue(submitted.await(5, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue(checks.get() >= 2);
            assertTrue(queries.get() >= 1);
        } finally { poller.close(); }
    }

    @Test
    void busyBuildDoesNotReadAndLeadershipLostDuringReadDoesNotPublish() {
        var builds = builds();
        var calls = new AtomicInteger();
        var leader = new java.util.concurrent.atomic.AtomicBoolean(true);
        var source = reader((k, l, t) -> {
            calls.incrementAndGet();
            leader.set(false);
            return CompletableFuture.completedFuture(List.of(new SidBucketClient.Row(k, "1", "C1C2")));
        });
        var poller = new IgraphConstraintTreePoller(source, builds, leader::get, "gul_item", true, true, 600, CLOCK);
        try {
            var ready = builds.getStatus();
            when(builds.getStatus()).thenReturn(new ConstraintTreeModels.BuildStatus(
                    ConstraintTreeModels.BuildState.BUILDING, 151, 150, 149, 1, 3, 1, 1, "building"));
            poller.pollOnce();
            assertEquals(0, calls.get());
            when(builds.getStatus()).thenReturn(ready);
            poller.pollOnce();
            assertEquals("FAILED", poller.getStatus().state());
            verify(builds, never()).submit(any());
        } finally { poller.close(); }
    }

    @Test
    void springEnvironmentVariablesActivateAndConfigurePoller() {
        var builds = builds();
        var leader = mock(org.flexlb.consistency.LBStatusConsistencyService.class);
        when(leader.isMaster()).thenReturn(true);
        var runner = new org.springframework.boot.test.context.runner.ApplicationContextRunner()
                .withUserConfiguration(IgraphConstraintTreePoller.class, org.flexlb.httpserver.IgraphConstraintTreeServer.class);
        runner.run(context -> {
            assertNull(context.getStartupFailure());
            assertTrue(context.getBeansOfType(IgraphConstraintTreePoller.class).isEmpty());
        });
        java.util.Map<String, Object> variables = java.util.Map.of(
                "CONSTRAINT_TREE_IGRAPH_ENABLED", "true", "CONSTRAINT_TREE_IGRAPH_MODEL", "gul_item",
                "CONSTRAINT_TREE_IGRAPH_KEY_PREFIX", "", "CONSTRAINT_TREE_IGRAPH_BUCKET_COUNT", "1",
                "CONSTRAINT_TREE_IGRAPH_BUCKET_ALGORITHM", "ITEM_ID_MOD",
                "CONSTRAINT_TREE_IGRAPH_SOURCE_READY", "true", "CONSTRAINT_TREE_IGRAPH_ALLOW_NON_ATOMIC_READ", "true");
        // ApplicationContextRunner registers conditional user classes before invoking initializers.
        // Install the environment in the context factory, matching normal boot startup ordering.
        new org.springframework.boot.test.context.runner.ApplicationContextRunner(() -> {
            var context = new org.springframework.context.annotation.AnnotationConfigApplicationContext();
            context.getEnvironment().getPropertySources().addFirst(
                    new org.springframework.core.env.SystemEnvironmentPropertySource("test-systemEnvironment", variables));
            return context;
        }).withUserConfiguration(IgraphConstraintTreePoller.class, org.flexlb.httpserver.IgraphConstraintTreeServer.class)
                .withBean(ConstraintTreeBuildService.class, () -> builds)
                .withBean(org.flexlb.consistency.LBStatusConsistencyService.class, () -> leader)
                .withBean(SidBucketClient.class, () -> (key, limit, timeout) -> CompletableFuture.completedFuture(
                        List.of(new SidBucketClient.Row(key, "1", "C1C2"))))
                .run(context -> {
                    assertNull(context.getStartupFailure());
                    var poller = context.getBean(IgraphConstraintTreePoller.class);
                    poller.pollOnce();
                    assertEquals("SUBMITTED", poller.getStatus().state());
                    assertEquals(1, poller.getStatus().buckets());
                    assertNotNull(context.getBean("igraphConstraintTreeRoutes"));
                });
    }
}
