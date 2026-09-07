# Rebalance profile migration and counter anchors

Reference: `a22f0678a2beb479c3da7ff9fa09df9c354f3d19`,
`elastic_rebalance` and `harness._run_batch`. BW and explicit SB/SN/WN variants
use the same corrected 12-stage program; only the expected consumer differs.
Other lifecycle variants remain explicitly restricted to BW and enqueue_batch.
The family-level queue requirement does not expand those variants.

Two cohorts of exactly 50 requests use ten concurrent workers, 2048 input / two
output tokens, three cold keys `[RID*100+j for j in range(3)]`, Schedule timeout
30s and stream timeout 15s. No warm-up or extra background traffic intervenes.
The second cohort begins after add/discovery/alive convergence and counter reads.
Both cohorts must finish without business errors. New accepted share remains
strictly positive and strictly below 0.6; there is no added positive old-worker
share floor. Complete records and protocol checks are explicit construction gates.

One BW counter anchor needed correction before expansion. Legacy computes old
worker deltas as `end - p_before - p_mid`, where `p_mid` is the baseline increment;
this is exactly `end - counter_at_baseline_end`. Thus old-worker counts accrued
during add/convergence belong to the measured denominator. Only the newcomer
baseline is captured after convergence. The previous candidate sampled all three
workers after convergence and could drop delayed old-worker accepts. Dedicated
`elastic_rebalance_anchor` combines the old snapshot from before add with the new
snapshot from after convergence. The artifact retains both source snapshots.
This restores the literal old window; it is not a change to the 60% threshold.

All four profiles are tested against the full old resolved configuration,
performance preset, 2P/4D topology, dynamic discovery, both cache pools and omitted
queue timeout. There is one dynamic Prefill addition and no birth configuration
override. The existing typed counter and exact topology checks remain declared
construction requirements; no Prefill batch-zero check claims route-owner cleanup.

`test_scenario_elastic_rebalance_profiles.py` runs actual RecordedRequests with
simulated RPCs. A ten-party barrier verifies simultaneous Schedule consumers in
each cohort. Schedule metadata selects Fetch or Generate; NON_BATCH reconstructs
the same shape and copies role addresses. Tests retain business and terminal/exit
records, reject successful streams using the wrong protocol, and keep business
errors independently failing. Boundary fixtures check 0/50 FAIL, 29/50 PASS and
30/50 FAIL. With ten delayed old accepts during add, 30 new accepts have denominator
60 and share 0.5, preserving the old counter anchor.

The batch helper's optional protocol guard defaults to None for all other
callers. The dedicated module's descriptors are aggregated through the existing
elastic_lifecycle handler list, so no shared elastic.py or catalog change is
required. Independent static and fixture review passed for
`18fa49dc3b8d5b3dba39bf29430e288a32f06932`: the reviewer independently ran 104
elastic tests in 22.232s and eight added-worker tests in 0.453s against core4e,
verified default-catalog compilation, and confirmed the seven non-rebalance
plans remain unchanged with exactly three new profile pairs. Real Java
acceptance remains pending. Legacy
cases and earlier result anchors remain available; those results do not establish
acceptance of this corrected BW program or its three new profiles.
