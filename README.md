# CloudMTP

**Cloud Multi-Token Prediction** — distributed speculative decoding across independent CPU machines.

## What is MTP?

Multi-Token Prediction (implemented here as *speculative decoding*) pairs a small, fast **draft model** with a large, strong **target model**. The draft autoregressively proposes a few candidate tokens; the target verifies all of them in a single parallel forward pass, accepts the prefix it agrees with, and corrects the first token it doesn't. Acceptance is governed by the target's own output distribution, so the output is statistically identical to the target model's — yet when the draft guesses well, several tokens are emitted per pass instead of one.

## Why it works: compute-bound vs. memory-bandwidth-bound

- **Prefill (encoding)** processes the whole prompt in parallel. Each weight loaded from RAM is reused across many token positions, so arithmetic intensity is high and the phase is **compute-bound**.
- **Decode** emits one token at a time. Every forward pass streams the entire weight matrix from RAM, uses each weight at a single token position, and discards it — arithmetic intensity is tiny, so the phase is **memory-bandwidth-bound**.

Verifying a batch of candidates looks like prefill, not decode. That moves target-model work back into the compute-bound regime and amortizes each weight load across multiple positions — the asymmetry CloudMTP exploits. Two further advantages: the draft only needs to predict often enough to keep acceptance high (not to match the target), and the two machines use separate memory bandwidth instead of competing for it.

## Architecture (iteration 1)

One draft worker, one checker, and a coordinator that performs no inference itself:

| Component | Where | Runs | Role |
|---|---|---|---|
| **Draft worker** | local machine | `gemma4:e2b` (llama.cpp) | Autoregressively drafts K candidates; owns its local KV cache |
| **MTP server** | server machine | Go, logic only | Routes batches, applies the acceptance policy, tracks round state, decides what the draft worker does next |
| **Checker worker** | server machine | `gemma4:e4b` (llama.cpp), 64k context | The "encoder": verifies a candidate batch in one parallel pass against its KV cache |

Only candidate token IDs and acceptance results cross the network; the checker keeps its own context and KV cache. Exact models, quantization, and stack are in [roadmap.md](roadmap.md).

## Decoding pipeline (one round, K = 4)

1. **Draft** — the draft worker decodes 4 candidate tokens from `gemma4:e2b`.
2. **Submit** — it sends the token IDs and current context position to the MTP server, which forwards the batch to the checker.
3. **Verify** — the checker runs one prefill-style pass over the candidates with `gemma4:e4b`, scoring every position in parallel. It returns which tokens were accepted, the first rejected position, the corrected token, and the target's own next-token prediction after the last accepted token — a free **bonus token** each round.
4. **Decide & respond** — the MTP server commits the accepted prefix, attaches the correction, and tells the draft worker how many tokens were accepted, the corrected token (if any), and the new position.
5. **Re-anchor & repeat** — the draft worker appends the accepted/corrected tokens, both caches roll back to the last accepted token if needed, and the next 4 candidates are drafted. Each round advances **k accepted + 1 bonus** tokens.

## Measuring it

| Configuration | Draft tokens |
|---|---|
| E4B alone (baseline) | 1 |
| E2B → E4B | 2, 4, 8, 16 |

```
accepted tokens / (draft time + verification time + network time)
```

compared against standalone E4B tokens/sec, with greedy decoding and a shared tokenizer. E4B verification time versus draft-token count is the decisive measurement: it shows how well weight loading amortizes and where the sweet spot sits on CPU hardware.

## Known challenges

| Challenge | Why it matters |
|---|---|
| Acceptance rate | Determines how much draft work is useful; the smaller the draft model, the harder this gets |
| Network latency | Fixed overhead on every round |
| KV cache management | The checker's cache must stay synchronized with accepted tokens |
| Memory capacity | 64k context KV cache plus weights is substantial RAM |
| Draft/target divergence | Long drafted sequences fall apart after the first rejection |

## Prior art

**[DSD — Decentralized Speculative Decoding](https://arxiv.org/abs/2511.11733)** (Gradient Network, 2025). Drafts a window of γ candidates, verifies in one synchronization round across nodes, and adds a training-free adaptive rule (τ-relaxed acceptance for low-impact tokens) worth an extra 15–20%; up to 2.56x on HumanEval, on sharded GPU clusters. *Takeaway:* they amortize cross-node sync of a sharded target; we run full models per machine, and our win is moving CPU decode from the memory-bound back into the compute-bound regime. We adopt their bonus token now and τ-relaxed acceptance later — iteration 1 is strict greedy.

**[Dovetail — CPU/GPU heterogeneous speculative decoding](https://github.com/ddInference/Dovetail)** (EMNLP 2025). Draft on a GPU, target verified on the CPU of the same machine; 1.79x–10.1x on 13B models. *Takeaway:* parallel CPU verification of a large model is lossless and fast, and on CPU **small draft windows win** (speedup declines past γ ≈ 16) — validating K = 4 and making the 2→16 sweep the key experiment. Same-family drafters reach high acceptance without training.

**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the inference engine for both workers; natively supports speculative decoding with a separate draft model.

## See also

- [roadmap.md](roadmap.md) — models, stack, and the phased plan
- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 (arXiv:2211.17192) — the original draft-then-verify formulation
