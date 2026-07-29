package org.flexlb.dao.netty;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Covers the claim/install handshake that decides who terminates an exchange, in particular the
 * window where a Netty thread ends it before the requesting thread has installed its sink.
 */
class HttpNettyChannelContextTest {

    private static FluxSink<String> sink() {
        AtomicReference<FluxSink<String>> captured = new AtomicReference<>();
        Flux.<String>create(captured::set).subscribe(v -> { }, e -> { });
        return captured.get();
    }

    @Test
    void claimBeforeTheSinkExistsRecordsTheCauseForWhoeverInstallsIt() {
        // The failing path wins the claim while sink == null, so it has nothing to terminate and
        // must hand its cause over — otherwise the real reason (read timeout, protocol error) is
        // dropped and the exchange is reported as a generic disconnect.
        HttpNettyChannelContext<String> ctx = new HttpNettyChannelContext<>();
        Throwable cause = new IllegalStateException("read timed out");

        assertNull(ctx.claimTermination(() -> cause),
                "no sink exists yet, so the claimant cannot terminate it itself");
        assertSame(cause, ctx.getPendingError(), "the cause must be kept for the installing thread");
        assertTrue(ctx.installSink(sink()),
                "the exchange already ended, so the installing thread owns the termination");
    }

    @Test
    void claimAfterTheSinkExistsTerminatesDirectlyAndRecordsNothing() {
        HttpNettyChannelContext<String> ctx = new HttpNettyChannelContext<>();
        FluxSink<String> installed = sink();

        assertFalse(ctx.installSink(installed), "the exchange is still live at install time");
        assertSame(installed, ctx.claimTermination(() -> new IllegalStateException("boom")),
                "the claimant gets the sink and fails it directly");
        assertNull(ctx.getPendingError(),
                "nothing to hand over when the claimant could terminate the sink itself");
    }

    @Test
    void onlyTheFirstClaimantOwnsTheTermination() {
        HttpNettyChannelContext<String> ctx = new HttpNettyChannelContext<>();
        Throwable first = new IllegalStateException("first");

        assertNull(ctx.claimTermination(() -> first));
        assertNull(ctx.claimTermination(() -> new IllegalStateException("second")),
                "a later path must not terminate an exchange that already ended");
        assertSame(first, ctx.getPendingError(), "the first claimant's cause is the one kept");
    }
}
