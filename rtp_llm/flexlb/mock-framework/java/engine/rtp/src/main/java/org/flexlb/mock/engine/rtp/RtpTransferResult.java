package org.flexlb.mock.engine.rtp;

import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.Scheduler;

record RtpTransferResult(int stopAfter, RequestFailure failure) implements Scheduler.TransferResult {}
