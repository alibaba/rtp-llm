package org.flexlb.mock.engine.service;

import org.flexlb.mock.engine.scheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.Scheduler;

public record TransferResult(boolean prefillFinished, RequestFailure failure) implements Scheduler.TransferResult {}
