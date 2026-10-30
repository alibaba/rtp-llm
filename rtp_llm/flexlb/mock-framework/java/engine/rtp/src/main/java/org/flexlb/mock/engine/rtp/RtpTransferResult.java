package org.flexlb.mock.engine.rtp;

import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.Scheduler;

record RtpTransferResult(boolean prefillFinished, RequestFailure failure) implements Scheduler.TransferResult {}
