# RTP-LLM Source Patches

This patch targets RTP-LLM commit
`d47f1b34cb129d0deb5dc2a04f9972ed374707bc`.

## Traffic simulation package

`rtp_llm_navi_sched_simulate.patch` builds the temporary online traffic-simulation
package. Apply it to a clean checkout:

```bash
git apply /path/to/navi_sched/adapters/rtp_llm/patches/rtp_llm_navi_sched_simulate.patch
```

This patch includes the required NaviSched package wiring. The resulting RTP
package has no mode switch:

- `FRONTEND` keeps RTP HTTP/tokenization/block hashing, uses NaviSched Client
  forward, and never creates `ModelRpcClient`;
- `PREFILL` runs NaviSched Agent with the test-only simulated executor;
- `PDFUSION` runs the formal NaviSched Master and RTP-compatible heartbeat HTTP;
- `DECODE` is rejected.

PREFILL and PDFUSION expose HTTP on `START_PORT` and Navi/Gig on
`START_PORT + 1`. The existing RTP startup command and deployment role selection
remain unchanged. See the patched `docs/backend/navi_sched.md` for package inputs,
service discovery, health endpoints, and validation commands.

The simulation patch is test-only. It must never be merged into a normal RTP model
serving release. Formal RTP integration is maintained directly on the local RTP
adapter branch so that it is reviewed and validated against the current source.
