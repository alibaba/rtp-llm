package org.flexlb.dispatcher;

import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import reactor.core.publisher.Mono;

import java.util.List;

/**
 * Resolves N BE targets in a single shot for the dispatcher's pre-assignment path. Returns
 * an empty list (never an error) when the resolver cannot service the call so the caller
 * can fall through to the no-pre-assignment path without try/catch noise.
 */
public interface BatchScheduleClient {

    Mono<List<BatchScheduleTarget>> requestTargets(int count);
}
