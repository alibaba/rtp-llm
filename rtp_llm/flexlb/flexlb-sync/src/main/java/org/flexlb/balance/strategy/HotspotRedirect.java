package org.flexlb.balance.strategy;

import org.flexlb.service.monitor.EngineHealthReporter;

/** Tracks one proposed redirect from its original worker through the Master's route commit. */
public final class HotspotRedirect implements AutoCloseable {
    private final EngineHealthReporter reporter;
    private final String sourceIp;
    private final String commitReason;
    private HotspotRedirectLimiter.Ticket ticket;
    private String abandonmentReason = "ROUTE_NOT_COMMITTED";
    private boolean finished;

    private HotspotRedirect(EngineHealthReporter reporter, String sourceIp, String commitReason) {
        this.reporter = reporter;
        this.sourceIp = sourceIp;
        this.commitReason = commitReason;
    }

    static HotspotRedirect observe(EngineHealthReporter reporter, PrefillCandidateSet candidates,
                                   PrefillCandidateRanking.Choice choice, boolean queued) {
        if (choice.hotspotSource() < 0) {
            if (choice.index() >= 0) {
                reporter.reportPrefillHotspotDecision(candidates.endpoint(choice.index()).getIp(),
                        "NOT_NEEDED", "ORDINARY_PLACEMENT");
            }
            return null;
        }
        String sourceIp = candidates.endpoint(choice.hotspotSource()).getIp();
        reporter.reportPrefillHotspotDecision(sourceIp, "NEEDED", "HOTSPOT");
        if (!choice.redirect()) {
            reporter.reportPrefillHotspotDecision(sourceIp, "FAILED", choice.noTargetReason());
            return null;
        }
        return new HotspotRedirect(reporter, sourceIp, queued ? "QUEUED_ROUTE_COMMITTED" : "ROUTE_COMMITTED");
    }

    void attach(HotspotRedirectLimiter.Ticket ticket) {
        this.ticket = ticket;
    }

    /** A provisional block may still be rescued. Only closing the route makes it a failure. */
    public void onAbandon(String reason) {
        abandonmentReason = reason;
    }

    public void commit() {
        if (finished) { return; }
        ticket.commit();
        finished = true;
        reporter.reportPrefillHotspotDecision(sourceIp, "SUCCESS", commitReason);
    }

    @Override
    public void close() {
        if (finished) { return; }
        finished = true;
        if (ticket != null) { ticket.close(); }
        reporter.reportPrefillHotspotDecision(sourceIp, "FAILED", abandonmentReason);
    }
}
