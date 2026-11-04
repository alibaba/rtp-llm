# Captured overload rounds

These are unmodified Master schedule replay records sampled from the
DeepSeek-VL service on 2026-10-22. They exercise overload admission followed
by the cost scheduler's restricted node view. They are current traffic
samples, not recordings of the historical 45-second incident.

| File | Requests | Nodes | Admission rejections | Optimizer dimensions |
| --- | ---: | ---: | ---: | ---: |
| overload-21416.json | 18 | 7 | 8 | 10 x 5 |
| overload-21457.json | 30 | 6 | 12 | 18 x 4 |

SHA-256 of the original JSON bytes:

```text
dacf3744cce5366be0a8c26adda933204acc64ea5588515ce89928eb28750dcb  overload-21416.json
50b92306b001f4b3d0e9031f20c65d44b551ef8959ded928d030e91fc969b114  overload-21457.json
```

Audit version 1 records node runtime inputs and random draws. Replay uses
the draws with the production admission and cold-start algorithms and
checks their computed outcomes against the recorded audit. The recorded
workload feature snapshot already includes progress up to schedule time.
