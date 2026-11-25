# CloudMTP Roadmap

## Models

| Role | Model | Quantization |
|---|---|---|
| Drafter (local) | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) | Q4_K_M |
| Decoder / checker (MTP server) | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/main) | Q4_K_M |

## Stack

Two custom programs, standard library only (`net/http`, `encoding/json`, `log/slog`, `testing`); everything else is stock llama.cpp:

- **`mtp-server` (orchestrator)** — Go. Owns all MTP logic: receives candidate batches from the agent, calls the checker's llama-server, applies the acceptance policy, decides what the agent does next. Performs no inference itself.
- **`mtp-agent` (draft client)** — Go, runs on the local machine next to the e2b llama-server. Owns the conversation context and chat template, drafts candidates, relays decisions, and runs the local benchmarks.
- **Transport** — plain HTTP request/response per round (`POST /verify` with JSON). The protocol is lock-step — the agent can't draft until it gets the decision — so request/response semantics fit exactly, and every round is curl-debuggable. Persistent JSON-lines over `net.Conn` is the stdlib upgrade path if per-request overhead ever shows up (it won't — model compute dominates by orders of magnitude).
- **Inference** — stock `llama-server` on both machines, no custom inference code. The checker exposes prompt-position logits via `echo=true` + `logprobs` on `/v1/completions` (requires an Oct 2026 or newer build). Do *not* enable `-md`/server-side speculation on the checker — that path currently returns placeholder logprobs (llama.cpp issue #27972).

```
mtp-agent ──HTTP/JSON── mtp-server ──HTTP/JSON── llama-server (e4b, checker)
     │
     └─HTTP/JSON── llama-server (e2b, drafter)
```

Zero third-party Go dependencies: the module builds offline, and both binaries cross-compile with `GOOS=linux GOARCH=amd64 go build`.

## Phase 1 — Foundations

- **Orchestration server:** build `mtp-server` in Go with basic responses; the speculative decoding level (K) and other settings are configured ahead of time, not tuned at runtime.
- **Automated benchmarking:** build a way to run automated LLM benchmarks and gather performance data. Runs on the draft device (the local machine hosting e2b).
- **Verifier plumbing:** ~~find a way to get encodings/logits from the checker model~~ — resolved: stock llama-server returns prompt-position top-k tokens via `echo=true` + `logprobs` on `/v1/completions`. One call verifies the whole candidate batch.

## Phase 2 — Speculative logic (offline)

- **Acceptance logic in the orchestration server:** compare proposed tokens against verifier output, keep the good tokens, discard the unused ones.
  - Unit tests covering the acceptance logic.
  - Not yet connected to actual model outputs — pure logic on fixtures.
- **Baseline benchmarks:** run base benchmarks and gather performance data on the models over a small benchmark set.
- **Transport smoke tests:** send tokens back and forth to the llama.cpp instances without wiring up the full flow.

## Phase 3 — Integration

- Connect the orchestration server to the models running in llama.cpp (draft + checker).
- Benchmark the models' end-to-end performance.

## Phase 4 — Performance tuning

- Adjust variables: draft length K, sampling settings, network cadence.
- **Performance tuning:** profile the end-to-end round (draft / network / verify breakdown), tune llama-server flags on both machines (threads, `-fa`, batch/ubatch sizes, KV cache quantization), and tune the orchestrator's HTTP handling against the key metric.

## Phase 5 — Multi-user support (optional)

- Multiple agents connected to one orchestrator concurrently: per-session context state, isolation between rounds.
- Scheduling multiple draft sessions against the single checker worker.
- Test other model pairings.

## Backlog (post-Phase 5)

- Pipeline parallelism.
- DSD-style adaptive verification: strict checks for key tokens, τ-relaxed acceptance for low-impact tokens (+15–20% over strict).
- Robustness: connection recovery, KV cache invalidation, long-context (64k) behavior.
