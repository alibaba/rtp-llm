# FlexLB TERM shutdown verification

## Shutdown behavior

`whale_start.sh` execs `startx.sh`, so its TERM handler runs in the container
entry PID. The supervisor synchronously calls `appctl.sh stop`, which sends
TERM to Java and waits for its exit before stopping auxiliary services.

Before Spring destroys serving resources, `ApplicationLifecycle` deregisters
the node and calls `FlexlbGrpcServer.drain()`:

1. Start a full quiet period **at shutdown**, regardless of how long the service
   was idle before TERM. Normal serving has no shutdown timer.
2. Each new Schedule RPC restarts the quiet period. The existing gRPC entry
   interceptor records a monotonic timestamp for both local and forwarded calls.
3. After a full quiet period, call gRPC `shutdown()` to stop accepting new calls,
   then `awaitTermination()` to wait for accepted calls to finish. The scheduler,
   forwarding channels and transport executors remain alive throughout this wait.
4. Only then let Spring destroy resources, Java exit, and the supervisor exit.

The quiet period is `grpcServer.shutdownQuietPeriodMs` in `FLEXLB_CONFIG`, default
**5000 ms**. Set it to `7000` in that JSON document to use seven seconds. It must
be a positive integer; the common config validator rejects invalid values during
startup. A change requires restarting the process. This replaces the application request counter
and per-request token bookkeeping; gRPC owns in-flight call tracking.

The explicit `/hook/pre_stop` compatibility endpoint uses the same drain and
returns success only after gRPC terminates. It does not have a separate internal
300s timeout. The TERM shell path does not call this HTTP hook. Automatic Spring
context closure also uses the drain path; normal idle serving never does.

The platform owns the **300s forced-kill deadline**. Java does not force completion
or destroy active requests on an internal timeout. `appctl.sh` waits up to
`FLEXLB_STOP_TIMEOUT_SECONDS` (default 300); if Java has not exited, stop reports
failure and the supervisor stays alive for platform termination. No early
SIGKILL is sent by these scripts.

Here “requests finish” means accepted FlexLB RPCs finish (including pending
Schedule dispatch/forwarding). Engine generation already handed off by a
successful Schedule can continue independently after FlexLB exits.

## Unit and transport regression tests

From the FlexLB directory:

```bash
./mvnw -B -P 'opensource,!internal' -pl flexlb-api,flexlb-mock-engine -am clean package \
  -Dtest=ConfigServiceTest,ApplicationLifecycleTest,FlexlbGrpcDrainTest,FlexlbServiceImplTest,AppStateHookServerTest,ScheduleForwardMatrixTest,FlexlbServiceCancelTest,FlexlbForwardHopGuardNettyTest,FollowerAsyncForwardingNettyTest \
  -Dsurefire.failIfNoSpecifiedTests=false
python3 -m unittest discover -s APP-META/docker-config/tests -v
```

The Netty tests use real TCP RPCs to verify normal idle, a full quiet period
starting at shutdown, repeated short arrivals resetting that period, and a
pending RPC surviving both quiet expiry and a shutdown-thread interrupt.
Spring tests verify drain finishes before serving resources are destroyed.
Shell tests verify stop failure blocks teardown/restart and first-start works.
The old HTTP token-holding fixture has been removed.

## Packaged application + mock engine + real entry TERM

```bash
python3 APP-META/docker-config/tests/run_mock_engine_term.py --scenario late --quiet-period-ms 7000 --output /tmp/flexlb-term-late
python3 APP-META/docker-config/tests/run_mock_engine_term.py --scenario queued --output /tmp/flexlb-term-queued
python3 APP-META/docker-config/tests/run_mock_engine_term.py --scenario engine --output /tmp/flexlb-term-engine
```

Output directories must not exist. Requires JDK 21, Python `grpcio`, `grpcio-tools`
and `protobuf`, as does the existing `flexlb_ft` harness. Build jars first using
the command above; do not rebuild them during a process test.

Every scenario launches `sh whale_start.sh`, the real supervisor and packaged
Spring Boot application, then sends SIGTERM to the **entry PID**. Full production
`appctl.sh stop` signals Java and waits. Only installation paths, ownership
preparation, Java launch setup and unused nginx/xagent/tomcat operations use a
local sandbox. Production Java components are not replaced.

An independent Java mock engine serves one prefill and one decode worker. Real
TCP gRPC follows Schedule → EnqueueBatch → FetchResponse:

- `late`: first idle longer than the configured quiet period without TERM; then
  send TERM and repeated short requests. All succeed and extend the quiet window.
- `queued`: occupy the sole decode slot with a long request, queue three more
  Schedule RPCs, and TERM the entry. Pending RPCs must survive beyond six seconds
  and finish before service exit, including after gRPC stops accepting new calls.
- `engine`: send TERM after Schedule has returned; FlexLB exits while independent
  engine generation continues and finishes successfully.

Each run checks request IDs, terminal output-length metadata, P/D completion
records, successful entry exit and shutdown log timestamps. It records client
and engine responses, process commands/PIDs, jar hashes and a result JSON.
Mock output metadata is checked; real model token contents are not generated.
These are single-master tests with consistency disabled. They do not exercise
ZK election, master handoff or production container cleanup.
