# CloudMTP

**Cloud Multi-Token Prediction** — distributed speculative decoding across independent CPU machines.

---

## What is MTP?

Multi-Token Prediction (implemented here as *speculative decoding*) is a technique for accelerating large language model inference by pairing two models: a small, fast **draft model** autoregressively generates a short sequence of candidate tokens, and a large, strong **target model** then verifies the entire candidate sequence in a single parallel forward pass, accepting the tokens it agrees with (and correcting the first one it doesn't). Because acceptance is governed by the target model's own output distribution, the generated text is statistically identical to what the target model would produce on its own yet whenever the draft model guesses well, the target emits several tokens per pass instead of one, substantially increasing throughput at zero quality cost.

## The Core Insight: Compute-Bound vs. Memory-Bandwidth-Bound

LLM inference has two phases with fundamentally different performance characteristics:

**Prefill (encoding)** — the prompt is processed in parallel. All token positions form large matrix multiplications where every weight loaded from memory is reused across hundreds or thousands of token positions. Arithmetic intensity (FLOPs per byte moved) is high, so this phase is **compute-bound**: more compute means proportionally faster prefill.

**Decode (generation)** — tokens are produced one at a time. Each forward pass fetches the entire weight matrix from RAM, uses every weight for only a couple of multiply-accumulates at a single token position, and discards it. Arithmetic intensity is tiny, so this phase is **memory-bandwidth-bound**: the hardware spends almost all its time streaming weights from memory rather than doing math.

This is exactly the asymmetry CloudMTP exploits: verification of a batch of candidate tokens looks like *prefill*, not *decode*. By moving decode work back into the compute-bound regime, each weight load is amortized across multiple token positions — turning wasted memory traffic into useful throughput.

## Architecture (Iteration 1)

Iteration 1 uses exactly **one draft worker** and **one checker worker**, plus a coordinator that handles all MTP logic but performs **no inference itself**:

```
┌────────────────────┐  candidates / results  ┌─────────────────────┐  candidates / results  ┌────────────────────┐
│    Draft Worker    │ ◄────────────────────► │     MTP Server      │ ◄────────────────────► │   Checker Worker   │
│   (local machine)  │                        │  (logic only —      │                        │  (server machine)  │
│                    │                        │   no inference)     │                        │                    │
│  gemma4:e2b        │                        │  routing            │                        │  llama.cpp         │
│  K-token drafting  │                        │  acceptance policy  │                        │  gemma4:e4b        │
│  local KV cache    │                        │  round state        │                        │  64k context       │
└────────────────────┘                        └─────────────────────┘                        └────────────────────┘
```

- **Draft worker (local machine)** — runs **gemma4:e2b** via llama.cpp. Autoregressively drafts candidate tokens and owns its local e2b KV cache.
- **MTP server (coordinator)** — pure logic, zero inference. Routes candidate batches to the checker, applies the acceptance policy, tracks round state, and decides what the draft worker does next. Lightweight enough to colocate with the checker machine to save a network hop.
- **Checker worker (server machine)** — runs llama.cpp with **gemma4:e4b** as the target model, **64k context**. This is the "encoder" side: it receives candidate sequences and verifies them in a single parallel pass against its KV cache, never decoding one token at a time.

## Decoding Pipeline

One high-level round-trip for a verification round (K = 4 draft tokens):

1. **Draft** — the draft worker autoregressively decodes **4 candidate tokens** from `gemma4:e2b` given the current shared context.
2. **Submit** — the draft worker sends the 4 candidate token IDs (plus the current context position) to the **MTP server**.
3. **Forward** — the MTP server forwards the candidate batch to the **checker worker**.
4. **Verify** — the checker runs one prefill-style forward pass over the candidates with `gemma4:e4b`, scoring every position in parallel against its KV cache. It returns per-position results to the MTP server: which tokens were accepted, the first rejected position, the corrected token (or logits) at that position, and the target's own next-token prediction after the last accepted position — a free bonus token per round.
5. **Decide** — the MTP server applies the acceptance policy: commit accepted tokens, attach the correction if there was a rejection, and choose the next instruction.
6. **Respond** — the server sends the outcome back to the draft worker: number of accepted tokens, the corrected token (if any), and the new context position.
7. **Re-anchor & repeat** — the draft worker appends accepted/corrected tokens to its context, both caches roll back to the last accepted token if needed, and it drafts the next 4 candidates. Each round advances **k accepted + 1 bonus** tokens. Loop until EOS or a length limit.

## Why This Should Work

1. **Verification batches amortize weight loads.** E4B decoding alone streams its weights once per token; verifying 8 candidates streams them once for 8 positions.
2. **Prefill-style processing has better arithmetic intensity.** More computation per byte moved moves E4B toward the compute-bound regime where CPUs are stronger.
3. **The draft model only needs decent accuracy**, not parity — E2B just has to predict often enough to keep acceptance rates high.
4. **The machines are independent.** Draft and target use separate memory bandwidth instead of competing for the same.

## Prototype Plan

- **Draft:** gemma4:e2b, local machine
- **Target:** gemma4:e4b via llama.cpp, MTP server
- **Context:** 64k
- **Decoding:** greedy initially
- **Shared tokenizer** between both models

### Benchmark Configurations

| Configuration | Draft tokens |
|---|---|
| E4B alone (baseline) | 1 |
| E2B → E4B | 2 |
| E2B → E4B | 4 |
| E2B → E4B | 8 |
| E2B → E4B | 16 |

### Key Metric

```
accepted tokens / (draft time + verification time + network time)
```

compared against ordinary standalone E4B tokens/sec. Also measure E4B verification time as the proposed-token count increases — this reveals how effectively weight loading is being amortized, and where the sweet spot sits on CPU hardware.

## Known Challenges

| Challenge | Why it matters |
|---|---|
| Acceptance rate | Determines how much draft work is useful; the smaller the draft model, the harder this gets |
| Network latency | Adds fixed overhead to every verification round |
| KV cache management | The target's cache must stay synchronized with accepted tokens |
| Memory capacity | 64k context KV cache plus model weights is substantial RAM |
| Draft/target divergence | Long drafted sequences fall apart after the first rejection |

## Prior Art

- **Decentralized Speculative Decoding (DSD)** — Gradient Network, arXiv:2511.11733 (Nov 2025). The closest work to CloudMTP. Drafts a window of γ candidates, verifies them in a single synchronization round across distributed nodes, and commits the accepted prefix at once. Adds a training-free *adaptive verification* rule (relaxation factor τ for low-impact tokens, strict checks for key tokens) worth an extra 15–20%. Reports up to 2.56x on HumanEval and 2.59x on GSM8K, ~37% communication reduction at 8 nodes — on sharded models, A800 GPUs, and 100 Gbps InfiniBand.
- **llama.cpp** — supports speculative decoding with a separate draft model, including independent thread and affinity controls for it.
- **Dovetail** — EMNLP 2025 (NUDT, pp. 17382–17395). CPU/GPU heterogeneous speculative decoding *within a single machine*: a custom-trained EAGLE-2-style draft head on a consumer GPU, with the target model (LLaMA2-Chat 7B/13B, Vicuna 13B) verified on the CPU. Cuts draft tokens to 16 to keep CPU verification latency low; 1.79x–10.1x speedups on consumer hardware needing only ~3GB VRAM. Code: `github.com/ddInference/Dovetail`.

### Relationship to DSD

Same core mechanism — draft K tokens, verify in one pass, commit the accepted prefix. The high-level takeaways for CloudMTP:

- **They shard the target model across nodes**, so every token costs a cross-node sync and their win is amortizing that. We run full models per machine, so the network only ever carries token IDs and acceptance results — and our regime is different: CPU decode is memory-bound, so our win is batching verification back into compute-bound territory.
- **What we adopt:** the free bonus token (the checker's verification pass also yields the target's own next-token prediction, so a round advances k+1 tokens) and, later, their τ-relaxed adaptive acceptance (+15–20%). Iteration 1 stays strict greedy.

### Relationship to Dovetail

Dovetail proves the core of our checker design: parallel CPU verification of a large model is lossless and fast (up to 10.1x on 13B). The high-level takeaways for CloudMTP:

- **Keep draft windows small on CPU** — verification time is the bottleneck, and their speedup declines past γ ≈ 16. Validates starting at K = 4 and makes our 2→16 sweep the decisive experiment.
- **Same-family drafters work without training** — their same-series drafter reached τ ≈ 4.9 from behavioral consistency with the target; ours is `gemma4:e2b`, off the shelf.
- **Their speedup formula is our key metric**: (γ·draft time + verify time) / (accepted tokens × standalone-e4b time per token).

## Roadmap

The phased implementation plan lives in [roadmap.md](roadmap.md).

## References

1. Song, J., Chen, W., Song, X., et al. **Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput.** Gradient Network, arXiv:2511.11733, Nov 2025.
2. Zhang, L., Zhang, Z., Xu, B., Li, R., Tian, Z., Mei, S., Li, D. **Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference.** EMNLP 2025, pp. 17382–17395. `github.com/ddInference/Dovetail`
3. Leviathan, Y., Kalman, M., Matias, Y. **Fast Inference from Transformers via Speculative Decoding.** ICML 2023, arXiv:2211.17192 — the original draft-then-verify formulation.
4. **llama.cpp** — speculative decoding implementation used by the draft and checker workers. `github.com/ggml-org/llama.cpp`
