package org.flexlb.mock.engine.service;

import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.Scheduler;

public record TransferResult(boolean prefillFinished, RequestFailure failure) implements Scheduler.TransferResult {}
