// CPU-only regression for native cancellation propagation across the prefill -> decode hop.
//
// What this pins. A prefill handler spends nearly all of its life blocked in client_stream->Read() on
// the prefill -> decode call, and the only isRequestCancelled() evaluations sit inside that read loop.
// When the downstream ClientContext was built independently of the upstream ServerContext, the two
// calls were unrelated as far as the gRPC core was concerned, so an upstream client cancellation could
// not reach a handler blocked in Read(): it stayed parked until the decode leg happened to emit another
// output. A decode that emits nothing -- still queued behind admission, or stalled -- never unblocked
// it, so the downstream call, its engine stream, its KV blocks and its per-rank admission slot were
// held for the whole generation and replacement requests queued behind them.
//
// ClientContext::FromServerContext makes the upstream server call the downstream call's C-core parent
// with GRPC_PROPAGATE_CANCELLATION (part of GRPC_PROPAGATE_DEFAULTS). When the upstream call receives
// its final op the core walks the parent's child list and cancels every inheriting child, which
// unblocks the pending Read() immediately -- no watcher thread, no polling.
//
// These tests drive a real in-process gRPC pair whose "decode" emits NO output at all, which is the
// worst case and the one that used to wedge. Pure gRPC + protobuf: no torch, no CUDA, no engine.

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace {

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;

// The acceptance target for reclaiming capacity: a cancellation must reach both legs and free them in
// well under a second while the peer is otherwise healthy.
constexpr int64_t kReleaseBudgetMs = 1000;

// ---------------------------------------------------------------------------
// The downstream ("decode") leg.
// ---------------------------------------------------------------------------

// Mirrors the shape of DecodeRpcServer::RemoteGenerate for the case that matters: it accepts the
// ALLOCATE request and then produces nothing. `emit_before_stall` lets a test model a stream that
// decoded a few rounds and then stalled, which is how a real ghost presents.
class SilentDecodeService final: public RpcService::Service {
public:
    ::grpc::Status RemoteGenerate(::grpc::ServerContext*                                            context,
                                  ::grpc::ServerReaderWriter<::GenerateOutputsPB, ::GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request)) {
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, "no allocate request");
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            allocate_received_at_ = Clock::now();
            allocate_received_    = true;
        }
        cv_.notify_all();

        for (int i = 0; i < emit_before_stall_; ++i) {
            if (context->IsCancelled() || stop_.load()) {
                break;
            }
            GenerateOutputsPB output;
            if (!stream->Write(output)) {
                break;
            }
        }

        // Emit nothing further: with no output, a cancellation check placed after a blocking output
        // wait never runs. Only the call's own cancellation can end this.
        while (!context->IsCancelled() && !stop_.load()) {
            std::this_thread::sleep_for(Ms(2));
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            cancelled_           = context->IsCancelled();
            released_at_         = Clock::now();
            downstream_released_ = true;
        }
        cv_.notify_all();
        return context->IsCancelled() ? ::grpc::Status(::grpc::StatusCode::CANCELLED, "downstream cancelled") :
                                        ::grpc::Status::OK;
    }

    // Test hooks -----------------------------------------------------------------
    void         requestStop() {
        stop_.store(true);
        cv_.notify_all();
    }
    void setEmitBeforeStall(int n) { emit_before_stall_ = n; }

    bool waitForAllocate(Ms timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return allocate_received_ || stop_.load(); });
    }
    bool waitForRelease(Ms timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return downstream_released_ || stop_.load(); });
    }
    bool sawCancellation() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cancelled_;
    }

private:
    mutable std::mutex        mu_;
    std::condition_variable   cv_;
    bool                      allocate_received_{false};
    bool                      downstream_released_{false};
    bool                      cancelled_{false};
    Clock::time_point         allocate_received_at_{};
    Clock::time_point         released_at_{};
    std::atomic<bool>         stop_{false};
    int                       emit_before_stall_{0};
};

// ---------------------------------------------------------------------------
// The upstream ("prefill") leg.
// ---------------------------------------------------------------------------

// Mirrors PrefillRpcServer::remoteAllocateResource + pollRemoteOutput: build the downstream context,
// open RemoteGenerate, send ALLOCATE, then block in Read(). `propagate` selects the two arms under
// test -- FromServerContext (shipped) versus an independent ClientContext (the pre-fix behaviour).
class FakePrefillService final: public RpcService::Service {
public:
    FakePrefillService(std::string decode_target, bool propagate):
        decode_target_(std::move(decode_target)), propagate_(propagate) {}

    ::grpc::Status GenerateStreamCall(::grpc::ServerContext*              context,
                                      const ::GenerateInputPB*            request,
                                      ::grpc::ServerWriter<::GenerateOutputsPB>* writer) override {
        (void)request;

        std::shared_ptr<::grpc::ClientContext> client_context;
        if (propagate_ && context != nullptr) {
            // Exactly the shipped construction: inherit cancellation, keep our own deadline policy.
            client_context = ::grpc::ClientContext::FromServerContext(
                *context, ::grpc::PropagationOptions().disable_deadline_propagation());
        } else {
            client_context = std::make_shared<::grpc::ClientContext>();
        }
        if (downstream_deadline_ms_ > 0) {
            client_context->set_deadline(std::chrono::system_clock::now() + Ms(downstream_deadline_ms_));
        }

        auto channel = ::grpc::CreateChannel(decode_target_, ::grpc::InsecureChannelCredentials());
        auto stub    = RpcService::NewStub(channel);
        auto stream  = stub->RemoteGenerate(client_context.get());

        GenerateRequestPB alloc_request;
        alloc_request.set_stage(RemoteStage::ALLOCATE);
        alloc_request.set_request_id(request_id_);
        if (!stream->Write(alloc_request)) {
            markReturned(::grpc::StatusCode::UNAVAILABLE, "downstream write failed");
            return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "downstream write failed");
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            blocked_in_read_ = true;
        }
        cv_.notify_all();

        // The blocked wait under test. A cancelled call must make this return false promptly.
        GenerateOutputsPB response;
        while (stream->Read(&response)) {
            if (!writer->Write(response)) {
                client_context->TryCancel();
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++outputs_forwarded_;
            }
            cv_.notify_all();
        }
        const auto status = stream->Finish();
        markReturned(status.error_code(), "downstream ended");
        return status;
    }

    // Test hooks -----------------------------------------------------------------
    void requestStop() {
        stop_.store(true);
        cv_.notify_all();
    }
    void setDownstreamDeadlineMs(int64_t ms) { downstream_deadline_ms_ = ms; }
    void setRequestId(int64_t id) { request_id_ = id; }

    bool waitForBlockedInRead(Ms timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return blocked_in_read_ || returned_ || stop_.load(); });
    }
    bool waitForReturn(Ms timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return returned_ || stop_.load(); });
    }
    bool waitForOutputs(int count, Ms timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(lock, timeout, [this, count] { return outputs_forwarded_ >= count || returned_; });
        return outputs_forwarded_ >= count;
    }
    bool         returned() const {
        std::lock_guard<std::mutex> lock(mu_);
        return returned_;
    }
    Clock::time_point returnedAt() const {
        std::lock_guard<std::mutex> lock(mu_);
        return returned_at_;
    }
    std::string returnDetail() const {
        std::lock_guard<std::mutex> lock(mu_);
        return return_detail_;
    }
    // Reset for the next request in a repeated-cancellation test.
    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        blocked_in_read_ = false;
        returned_        = false;
        outputs_forwarded_ = 0;
        return_detail_.clear();
    }

private:
    void markReturned(::grpc::StatusCode code, const std::string& detail) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            returned_       = true;
            returned_at_    = Clock::now();
            return_detail_  = detail + " (grpc code " + std::to_string(static_cast<int>(code)) + ")";
        }
        cv_.notify_all();
    }

    std::string             decode_target_;
    bool                    propagate_;
    std::atomic<bool>       stop_{false};
    std::atomic<int64_t>    downstream_deadline_ms_{0};
    std::atomic<int64_t>    request_id_{1};
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    bool                    blocked_in_read_{false};
    bool                    returned_{false};
    int                     outputs_forwarded_{0};
    Clock::time_point       returned_at_{};
    std::string             return_detail_;
};

// ---------------------------------------------------------------------------
// Harness: one in-process decode server + one in-process prefill server.
// ---------------------------------------------------------------------------

class PdPair {
public:
    PdPair(bool propagate, int emit_before_stall = 0) {
        decode_.setEmitBeforeStall(emit_before_stall);

        ::grpc::ServerBuilder decode_builder;
        int                     decode_port = 0;
        decode_builder.AddListeningPort("127.0.0.1:0", ::grpc::InsecureServerCredentials(), &decode_port);
        decode_builder.RegisterService(&decode_);
        decode_server_ = decode_builder.BuildAndStart();
        if (!decode_server_) {
            init_error_ = "decode server failed to start";
            return;
        }
        decode_target_ = "127.0.0.1:" + std::to_string(decode_port);

        prefill_ = std::make_unique<FakePrefillService>(decode_target_, propagate);

        ::grpc::ServerBuilder prefill_builder;
        int                   prefill_port = 0;
        prefill_builder.AddListeningPort("127.0.0.1:0", ::grpc::InsecureServerCredentials(), &prefill_port);
        prefill_builder.RegisterService(prefill_.get());
        prefill_server_ = prefill_builder.BuildAndStart();
        if (!prefill_server_) {
            init_error_ = "prefill server failed to start";
            return;
        }
        prefill_target_ = "127.0.0.1:" + std::to_string(prefill_port);

        channel_ = ::grpc::CreateChannel(prefill_target_, ::grpc::InsecureChannelCredentials());
        stub_    = RpcService::NewStub(channel_);
    }

    ~PdPair() {
        // Deterministic teardown: unblock both fake handlers before joining the servers, so a test
        // that leaves the prefill parked in Read() (the non-propagating control arm) cannot hang.
        decode_.requestStop();
        if (prefill_) {
            prefill_->requestStop();
        }
        if (prefill_server_) {
            prefill_server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        }
        if (decode_server_) {
            decode_server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        }
    }

    // Start one upstream GenerateStreamCall on its own thread; returns the thread. An optional upstream
    // deadline lets a test compare the two hops' budgets in either direction.
    std::thread startUpstream(std::shared_ptr<::grpc::ClientContext>* out_ctx, int64_t upstream_deadline_ms = 0) {
        auto ctx = std::make_shared<::grpc::ClientContext>();
        if (upstream_deadline_ms > 0) {
            ctx->set_deadline(std::chrono::system_clock::now() + Ms(upstream_deadline_ms));
        }
        *out_ctx = ctx;
        return std::thread([this, ctx]() {
            GenerateInputPB input;
            input.set_request_id(1);
            reader_ = stub_->GenerateStreamCall(ctx.get(), input);
            GenerateOutputsPB out;
            while (reader_->Read(&out)) {
            }
            upstream_status_ = reader_->Finish();
        });
    }

    void                 joinUpstream(std::thread& t) {
        if (t.joinable()) {
            t.join();
        }
    }
    ::grpc::Status       upstreamStatus() const { return upstream_status_; }

    FakePrefillService*  prefill() { return prefill_.get(); }
    SilentDecodeService* decode() { return &decode_; }

    // A constructor cannot use ASSERT_*, so startup failures are reported here instead.
    bool                ok() const { return init_error_.empty(); }
    const std::string&  initError() const { return init_error_; }

private:
    std::string                             init_error_;
    SilentDecodeService                     decode_;
    std::unique_ptr<::grpc::Server>         decode_server_;
    std::string                             decode_target_;
    std::unique_ptr<FakePrefillService>     prefill_;
    std::unique_ptr<::grpc::Server>         prefill_server_;
    std::string                             prefill_target_;
    std::shared_ptr<::grpc::Channel>        channel_;
    std::unique_ptr<RpcService::Stub>       stub_;
    std::unique_ptr<::grpc::ClientReader<GenerateOutputsPB>> reader_;
    ::grpc::Status                          upstream_status_;
};

int64_t msSince(Clock::time_point t0) {
    return std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// 1. The propagation policy is what we intend, independent of any timing.
// ---------------------------------------------------------------------------

TEST(PropagationPolicy, DefaultsCarryCancellationAndTheExplicitDeadlinePolicyIsPreserved) {
    ::grpc::PropagationOptions defaults;
    // Cancellation propagation is the whole point, and it is on by default.
    EXPECT_NE(defaults.c_bitmask() & GRPC_PROPAGATE_CANCELLATION, 0u);

    // Disabling deadline propagation must not disturb cancellation: the core would otherwise take
    // GPR_MIN(our deadline, the parent's) and silently replace the explicit downstream policy with
    // the upstream call's deadline.
    ::grpc::PropagationOptions opts;
    opts.disable_deadline_propagation();
    EXPECT_EQ(opts.c_bitmask() & GRPC_PROPAGATE_DEADLINE, 0u);
    EXPECT_NE(opts.c_bitmask() & GRPC_PROPAGATE_CANCELLATION, 0u);
}

// ---------------------------------------------------------------------------
// 2. The regression itself: a silent downstream must still be released.
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, UpstreamCancelReleasesASilentDownstreamWithinBudget) {
    PdPair pair(/*propagate=*/true);
    ASSERT_TRUE(pair.ok()) << pair.initError();

    std::shared_ptr<::grpc::ClientContext> ctx;
    auto                                   client = pair.startUpstream(&ctx);

    // Both legs are now parked: the prefill inside Read(), the decode emitting nothing.
    ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10)))
        << "prefill never reached the downstream Read()";
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)))
        << "decode never received the ALLOCATE request";

    const auto cancel_at = Clock::now();
    ctx->TryCancel();

    // The prefill handler must come back out of Read() on its own, and the downstream call must see
    // its own cancellation -- that is what releases the engine stream, KV and admission slot.
    ASSERT_TRUE(pair.prefill()->waitForReturn(Ms(kReleaseBudgetMs)))
        << "prefill stayed blocked in Read() after the upstream cancel (detail: "
        << pair.prefill()->returnDetail() << ")";
    const int64_t prefill_ms = std::chrono::duration_cast<Ms>(pair.prefill()->returnedAt() - cancel_at).count();

    ASSERT_TRUE(pair.decode()->waitForRelease(Ms(kReleaseBudgetMs)))
        << "downstream never observed its own cancellation";

    EXPECT_TRUE(pair.decode()->sawCancellation()) << "downstream exited without a cancellation";
    EXPECT_LT(prefill_ms, kReleaseBudgetMs) << "prefill release took " << prefill_ms << " ms";
    EXPECT_LT(msSince(cancel_at), kReleaseBudgetMs);

    pair.joinUpstream(client);
}

TEST(CancellationPropagation, ReleaseAlsoHoldsWhenTheDownstreamStallsMidStream) {
    // A stream that decoded a few rounds and then stalled: outputs exist, then stop. Cancellation
    // must not depend on another output arriving.
    PdPair pair(/*propagate=*/true, /*emit_before_stall=*/3);
    ASSERT_TRUE(pair.ok()) << pair.initError();

    std::shared_ptr<::grpc::ClientContext> ctx;
    auto                                   client = pair.startUpstream(&ctx);

    ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10)));
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));
    ASSERT_TRUE(pair.prefill()->waitForOutputs(3, std::chrono::seconds(10)));
    ASSERT_FALSE(pair.prefill()->waitForReturn(Ms(100)))
        << "prefill returned before cancellation instead of waiting through the mid-stream stall";

    const auto cancel_at = Clock::now();
    ctx->TryCancel();

    ASSERT_TRUE(pair.prefill()->waitForReturn(Ms(kReleaseBudgetMs)))
        << "prefill stayed blocked after a mid-stream stall (detail: " << pair.prefill()->returnDetail() << ")";
    ASSERT_TRUE(pair.decode()->waitForRelease(Ms(kReleaseBudgetMs)));
    EXPECT_TRUE(pair.decode()->sawCancellation());
    EXPECT_LT(msSince(cancel_at), kReleaseBudgetMs);

    pair.joinUpstream(client);
}

// ---------------------------------------------------------------------------
// 3. The control arm: without the link the handler stays parked. This is what
//    makes the test above meaningful rather than a tautology.
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, WithoutTheLinkTheHandlerStaysBlocked) {
    PdPair pair(/*propagate=*/false);
    ASSERT_TRUE(pair.ok()) << pair.initError();

    std::shared_ptr<::grpc::ClientContext> ctx;
    auto                                   client = pair.startUpstream(&ctx);

    ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10)));
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));

    ctx->TryCancel();

    // The downstream is unrelated to the upstream call, so nothing wakes the parked Read(): this is
    // the pre-fix behaviour the production change removes. Bounded so the test cannot hang.
    EXPECT_FALSE(pair.prefill()->waitForReturn(Ms(1500)))
        << "the independent ClientContext released on its own; the propagation link may not be what "
           "this suite is measuring (detail: "
        << pair.prefill()->returnDetail() << ")";
    EXPECT_FALSE(pair.decode()->sawCancellation());

    pair.decode()->requestStop();
    pair.prefill()->requestStop();
    pair.joinUpstream(client);
}

// ---------------------------------------------------------------------------
// 4. Deadline policy: the explicit downstream deadline still governs.
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, ExplicitDownstreamDeadlineStillAppliesWithoutAClientCancel) {
    PdPair pair(/*propagate=*/true);
    ASSERT_TRUE(pair.ok()) << pair.initError();
    pair.prefill()->setDownstreamDeadlineMs(400);

    std::shared_ptr<::grpc::ClientContext> ctx;
    auto                                   client = pair.startUpstream(&ctx);

    ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10)));
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));

    // Nobody cancels. The downstream must end on its OWN explicit deadline, proving that disabling
    // deadline propagation left set_deadline() authoritative rather than overriding it.
    const auto t0 = Clock::now();
    ASSERT_TRUE(pair.prefill()->waitForReturn(std::chrono::seconds(5)))
        << "the explicit downstream deadline did not end the call (detail: "
        << pair.prefill()->returnDetail() << ")";
    const int64_t elapsed = msSince(t0);
    EXPECT_GE(elapsed, 300) << "ended before the 400 ms deadline: " << elapsed << " ms";
    EXPECT_LE(elapsed, 2500) << "deadline took far longer than configured: " << elapsed << " ms";

    pair.joinUpstream(client);
}

// ---------------------------------------------------------------------------
// 5. Normal completion is unaffected.
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, NormalCompletionIsUnaffected) {
    // The decode emits outputs and then finishes instead of stalling; the prefill must relay and
    // return normally, so the propagation link does not perturb the happy path.
    PdPair pair(/*propagate=*/true, /*emit_before_stall=*/0);
    ASSERT_TRUE(pair.ok()) << pair.initError();
    pair.prefill()->setDownstreamDeadlineMs(0);

    std::shared_ptr<::grpc::ClientContext> ctx;
    // Use a fresh pair whose decode completes: model that by stopping the decode loop after the
    // allocate, which makes Read() return false and the prefill finish normally.
    auto client = pair.startUpstream(&ctx);
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));
    pair.decode()->requestStop();

    ASSERT_TRUE(pair.prefill()->waitForReturn(std::chrono::seconds(5)));
    EXPECT_FALSE(pair.decode()->sawCancellation())
        << "a normal shutdown was reported as a cancellation";
    pair.joinUpstream(client);
    EXPECT_TRUE(pair.upstreamStatus().ok());
}

// ---------------------------------------------------------------------------
// 6. Repeated cancellations must each release, not just the first.
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, RepeatedCancellationsEachRelease) {
    for (int round = 0; round < 3; ++round) {
        PdPair pair(/*propagate=*/true);
        ASSERT_TRUE(pair.ok()) << pair.initError();
        pair.prefill()->setRequestId(round + 1);

        std::shared_ptr<::grpc::ClientContext> ctx;
        auto                                   client = pair.startUpstream(&ctx);

        ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10))) << "round " << round;
        ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10))) << "round " << round;

        const auto cancel_at = Clock::now();
        ctx->TryCancel();

        ASSERT_TRUE(pair.prefill()->waitForReturn(Ms(kReleaseBudgetMs)))
            << "round " << round << " did not release (detail: " << pair.prefill()->returnDetail() << ")";
        ASSERT_TRUE(pair.decode()->waitForRelease(Ms(kReleaseBudgetMs))) << "round " << round;
        EXPECT_TRUE(pair.decode()->sawCancellation()) << "round " << round;
        EXPECT_LT(msSince(cancel_at), kReleaseBudgetMs) << "round " << round;

        pair.joinUpstream(client);
    }
}

// ---------------------------------------------------------------------------
// 7. The two hops' deadlines are independent in BOTH directions (deadline propagation stays disabled
//    while cancellation propagation stays enabled).
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, UpstreamDeadlineEarlierThanDownstreamStillReleasesTheChild) {
    PdPair pair(/*propagate=*/true);
    ASSERT_TRUE(pair.ok()) << pair.initError();
    // The downstream policy is far looser than the upstream budget. If the child ended on its OWN
    // deadline it would take 5 s; it must instead be released when the upstream call expires. Because
    // deadline propagation is disabled, that release cannot be inheritance -- it is the parent's final
    // op cascading cancellation to the child, which is exactly what unblocks a handler parked in Read().
    pair.prefill()->setDownstreamDeadlineMs(5000);

    std::shared_ptr<::grpc::ClientContext> ctx;
    auto                                   client = pair.startUpstream(&ctx, /*upstream_deadline_ms=*/400);

    ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10)));
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));

    const auto t0 = Clock::now();
    ASSERT_TRUE(pair.prefill()->waitForReturn(std::chrono::seconds(10)))
        << "an expired upstream call did not release the downstream (detail: "
        << pair.prefill()->returnDetail() << ")";
    const int64_t elapsed = msSince(t0);
    EXPECT_GE(elapsed, 300) << "released before the upstream deadline expired: " << elapsed << " ms";
    EXPECT_LT(elapsed, 4000)
        << "released at/near the 5 s downstream deadline, so this was deadline inheritance or a stall "
           "rather than cancellation propagation: "
        << elapsed << " ms";
    // The downstream leg must be released too, not merely the prefill's blocked read. Wait for it: the
    // decode records its cancellation only as it exits its own poll loop.
    ASSERT_TRUE(pair.decode()->waitForRelease(Ms(kReleaseBudgetMs)))
        << "an expired upstream call released the prefill but left the downstream leg held";
    EXPECT_TRUE(pair.decode()->sawCancellation())
        << "the downstream did not observe the cancellation that released it";

    pair.joinUpstream(client);
}

TEST(CancellationPropagation, DownstreamDeadlineEarlierThanUpstreamEndsTheChildFirst) {
    PdPair pair(/*propagate=*/true);
    ASSERT_TRUE(pair.ok()) << pair.initError();
    pair.prefill()->setDownstreamDeadlineMs(400);

    std::shared_ptr<::grpc::ClientContext> ctx;
    // The upstream budget is far longer, so the tighter EXPLICIT downstream deadline must still govern.
    // Together with the case above this pins both directions: neither hop's budget silently replaces
    // the other's, which is what disabling deadline propagation is supposed to guarantee.
    auto client = pair.startUpstream(&ctx, /*upstream_deadline_ms=*/5000);

    ASSERT_TRUE(pair.prefill()->waitForBlockedInRead(std::chrono::seconds(10)));
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));

    const auto t0 = Clock::now();
    ASSERT_TRUE(pair.prefill()->waitForReturn(std::chrono::seconds(10)))
        << "the tighter downstream deadline did not end the call (detail: "
        << pair.prefill()->returnDetail() << ")";
    const int64_t elapsed = msSince(t0);
    EXPECT_GE(elapsed, 300) << "ended before its own 400 ms deadline: " << elapsed << " ms";
    EXPECT_LT(elapsed, 2500) << "downstream deadline took far longer than configured: " << elapsed << " ms";

    pair.joinUpstream(client);
    // The handler forwards the downstream status, so DEADLINE_EXCEEDED alone cannot distinguish
    // which hop expired. The upstream deadline must still be in the future when the child ends.
    EXPECT_LT(std::chrono::system_clock::now(), ctx->deadline());
    EXPECT_EQ(::grpc::StatusCode::DEADLINE_EXCEEDED, pair.upstreamStatus().error_code());
}

// ---------------------------------------------------------------------------
// 8. The no-server-context path (a deferred caller) keeps working.
// ---------------------------------------------------------------------------

TEST(CancellationPropagation, NullParentFallbackPathStillCompletesNormally) {
    // Production builds an independent ClientContext when there is no server context to inherit from:
    // the deferred batch path constructs slot contexts on a worker pool thread after the originating
    // handler has returned, and the core asserts that a propagation parent is a server call. This is
    // that same construction, and it must serve a normal request -- not merely "fail to propagate".
    PdPair pair(/*propagate=*/false);
    ASSERT_TRUE(pair.ok()) << pair.initError();
    pair.prefill()->setDownstreamDeadlineMs(0);

    std::shared_ptr<::grpc::ClientContext> ctx;
    auto                                   client = pair.startUpstream(&ctx, /*upstream_deadline_ms=*/5000);
    ASSERT_TRUE(pair.decode()->waitForAllocate(std::chrono::seconds(10)));
    pair.decode()->requestStop();  // the downstream ends normally

    ASSERT_TRUE(pair.prefill()->waitForReturn(std::chrono::seconds(10)))
        << "the fallback context did not complete a normal request (detail: "
        << pair.prefill()->returnDetail() << ")";
    EXPECT_FALSE(pair.decode()->sawCancellation())
        << "a normal downstream shutdown was reported as a cancellation on the fallback path";

    pair.joinUpstream(client);
    EXPECT_TRUE(pair.upstreamStatus().ok());
}

}  // namespace
