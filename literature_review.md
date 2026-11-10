# ShardFit — Related Work & Literature Review

**Repository:** https://github.com/Abhay27112006/Shardfit  
**Branch:** master  
**Purpose:** Survey of prior art related to ShardFit's architecture and the supersphere router concept, with comparisons to the proposed idea.

---

## Table of Contents

1. [Related Papers (Oldest to Newest)](#1-related-papers-oldest-to-newest)
2. [ShardFit Core Architecture](#2-shardfit-core-architecture)
3. [Supersphere Router Idea](#3-supersphere-router-idea)
4. [Similarities with Prior Work](#4-similarities-with-prior-work)
5. [Differences from Prior Work](#5-differences-from-prior-work)
6. [Honest Assessment](#6-honest-assessment)

---

## 1. Related Papers (Oldest to Newest)

| Year | Paper | Authors | Core Idea | Relevance to ShardFit |
|------|-------|---------|-----------|----------------------|
| **1991** | [Adaptive Mixtures of Local Experts](https://www.mitpressjournals.org/doi/10.1162/neco.1991.3.1.79) | Jacobs, Jordan, Nowlan, Hinton | Introduces Mixture of Experts (MoE): a gating network routes inputs to specialized "expert" networks. Each expert learns a subset of the data. | Foundational for router-based expert selection — ancestor of router + worker architectures. |
| **2003** | [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) | Bengio, Ducharme, Vincent, Jauvin | One of the first neural language models — maps words to continuous vectors and predicts the next word using a feedforward network over embeddings. | Shows words can be represented as continuous vectors — conceptual predecessor to geometric word representations. |
| **2013** | [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546) (word2vec) | Mikolov, Chen, Corrado, Dean | Learns word vectors by predicting surrounding words (skip-gram) or center words from context (CBOW). Words with similar meaning cluster in vector space; cosine similarity measures semantic similarity. | Directly analogous to supersphere "word kets" — each word is a vector; cosine similarity measures relatedness. |
| **2014** | [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) | Pennington, Socher, Manning | Combines global matrix factorization with local context windows. Word vectors trained so dot product of two words approximates log co-occurrence probability. | Another vector-space word model showing geometric relationships capture semantic meaning. |
| **2016** | [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) (FastText) | Bojanowski, Grave, Joulin, Mikolov | Represents words as bags of character n-grams, enabling construction of vectors for out-of-vocabulary words. | Shows subword structure can be embedded geometrically — relevant for handling unseen words in geometric models. |
| **2017** | [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) | Gehring, Auli, Grangier, Dauphin, Auli | Shows convolutional architectures can compete with Transformers for sequence modeling, with parallelizable computation. | Proves non-Transformer architectures can do language modeling — supersphere is another non-Transformer approach. |
| **2017** | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean | Applies MoE to Transformer-scale models — thousands of expert FFN layers, a router selects top-k experts per token. Scales to massive model sizes while keeping compute constant. | Closest modern MoE predecessor — routes tokens to expert sub-layers within one model. |
| **2021** | [RETRO: Improving Language Models by Retrieving Trillions of Tokens](https://arxiv.org/abs/2112.04426) | Lewis, Bhosale, Timokhin, Goyal, Mohamed, et al. | Retrieval-enhanced language model that fetches relevant chunks from a large database and conditions generation on them, rather than storing everything in parameters. | "Don't run everything at once" — retrieve only what's needed, similar to loading only relevant model chunks. |
| **2022** | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Leviathan, Kalman, Matias | A small, fast draft model proposes multiple tokens; a large target model verifies them in parallel. Accepted tokens are kept, rejected ones discarded. | Direct ancestor of "small model proposes, large model verifies" — small-fast + big-verify pattern. |
| **2022** | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Sparsely Activated Routing](https://arxiv.org/abs/2101.03961) | Fedus, Zoph, Shazeer | Simplifies MoE routing to top-1 expert per token (instead of top-k), reducing communication overhead. Scales to trillion parameters. | Shows extremely sparse routing works at massive scale — your "one SLM per prompt" is an extreme version. |
| **2023** | [Speculative Inference: Accelerating Large Language Models via Drafting and Verification](https://arxiv.org/abs/2302.01318) | Chen, Borgeaud, Irving, et al. | Generalizes speculative decoding — various drafting strategies (smaller model, earlier checkpoint) paired with verification by a larger model. | Reinforces small-fast + large-verify pattern your supervisor LLM implements. |
| **2023** | [FiRST / D-LLM / DASH — Dynamic Layer Skipping](https://arxiv.org/abs/2302.05992) | Various authors | Skip transformer layers dynamically based on input complexity, or exit early from shallow layers for easy inputs. | Mentioned in ShardFit's non-goals (Section 15): ShardFit does NOT skip layers. |

---

## 2. ShardFit Core Architecture

ShardFit is a runtime that lets one small GPU run open-source language models larger than its memory, **losslessly**, by splitting the model into shards and streaming them through VRAM.

| Component | Description |
|-----------|-------------|
| **Shards** | Contiguous blocks of transformer layers (or attention/MLP sub-blocks), stored as one contiguous buffer so transfer is one large copy. |
| **VRAM Budget** | V = total VRAM; R = reserved for KV cache/activations; W = V − R = weights budget; S = streaming slots (≥2); C = slot size; Pin = W − S·C = resident shard budget. |
| **Execution Loop** | Compute stream runs shard *i* while copy stream loads shard *i+1* into a free slot (non-blocking copies, CUDA events). Slots rotate. |
| **Storage Tiers** | GPU (pinned) → host RAM (pinned) → NVMe. Scheduler assigns tiers by measured bandwidth. |
| **Lossless** | Output must match a reference implementation running the same weights. Only approved approximation is quantization, always disclosed. |
| **No Skipping** | Every shard runs on every pass. No layer skipping, no shard skipping, no routers, no bridge networks. |

---

## 3. Supersphere Router Idea

### Core Concept

Instead of one monolithic LLM, use an architecture where:

1. **Supersphere geometry** — Each domain has its own N-dimensional unit sphere (S^{N-1} in R^N). Default dimension: 128. All vectors are unit length (||v||₂ = 1).
2. **Word kets** — Each vocabulary word has a trainable unit vector ("ket") on the sphere. Kets are initialized randomly and evolve during training.
3. **Context ket** — Built from preceding word kets using exponential decay weighting: recent words get higher weight. Result projected back onto the sphere.
4. **Angular scoring** — Given a context ket, compare it with all word kets using cosine similarity (dot product, since both are unit vectors). Convert scores to probabilities via softmax with temperature.
5. **Domain-specific SLMs** — Tiny language models (1B–3B parameters), each specialized for one domain (code, math, creative writing, etc.), small enough to run on CPUs like Intel Celeron.
6. **Supersphere router** — For an incoming prompt, build its context ket and score it against each domain's sphere using cosine similarity. The domain with highest similarity wins.
7. **Supervisor LLM** — A bigger model that reviews SLM outputs and overrides them when router confidence is below a threshold.

### Mathematical Formulation

**Supersphere:** S^{N-1} = {x ∈ R^N : ||x||₂ = 1}

**Context construction:**
c_t = normalize(Σ_{i=1}^{L} α^{L−i} · k_{wᵢ})

where α ∈ (0,1) is the decay factor, k_{wᵢ} is the ket for word wᵢ, and L is context length.

**Angular scoring:**
score(w) = cosine_similarity(context, ket_w) = context · ket_w (since both are unit vectors)

**Prediction:**
P(w\|context) = softmax(score(w) / temperature)

### Training

Training uses predictive evolution of word kets:
- For each training sequence, predict next word from context
- Compute cross-entropy loss
- Update target word ket by moving it along the sphere tangent space (geodesic update)
- Retract back to the sphere after each update

---

## 4. Similarities with Prior Work

| Your Idea | Similar Paper(s) | Similarity |
|-----------|-----------------|------------|
| Words as vectors on a sphere (word kets) | word2vec (2013), GloVe (2014), FastText (2016) | Words represented as geometric vectors; cosine similarity measures semantic relatedness |
| Router selects specialized sub-network per input | MoE (1991), Switch Transformers (2022), Shazeer et al. (2017) | A routing mechanism chooses which expert/component handles each input |
| Domain-specific SLMs | Switch Transformers (2022), MoE (1991) | Sparsely activated specialized sub-networks handle different parts of the input space |
| Supervisor LLM reviews/corrects SLM output | Speculative Decoding (2022), Speculative Inference (2023) | Small model proposes, large model verifies/corrects |
| No attention / non-Transformer | ConvSeq2Seq (2017), RETRO (2021) | Non-Transformer architectures can perform language modeling |
| SLMs on weak hardware (Intel Celeron) | Edge/on-device ML literature | Small models can run on constrained hardware with limited RAM/GPU |
| Context as weighted sum of previous word vectors | word2vec CBOW (2013) | Context represented as aggregation of word vectors |

---

## 5. Differences from Prior Work

| Your Idea | Prior Work | Key Difference |
|-----------|-----------|---------------|
| Vectors constrained to unit sphere (||v||₂ = 1) | word2vec / GloVe | Word2Vec/GloVe vectors live in free R^N space. Your kets are constrained to the unit sphere and trained with spherical gradient updates (retract to tangent space). |
| Router uses geometric cosine similarity | MoE / Switch Transformers (2017, 2022) | MoE routers are learned neural networks trained end-to-end with backprop. Your router is purely geometric — cosine similarity between context ket and domain sphere, no learned gating parameters. |
| Routes entire prompts to separate SLM models | MoE layers route tokens to expert sub-layers | MoE routes individual tokens to expert sub-layers *within one model*. Your router routes entire prompts to *entirely separate SLM models* with their own vocabularies and weights. |
| Supervisor reviews complete outputs | Speculative Decoding verifies tokens | Speculative decoding verifies individual tokens within one forward pass. Your supervisor reviews complete outputs from separate models, potentially with different vocabularies or domains. |
| No attention, no transformer layers | All modern LLMs (Transformers) | Your supersphere has no self-attention, no feed-forward layers, no positional encoding. It is a geometric nearest-neighbor predictor, not a deep network. |
| Trained per domain with separate spheres | Single monolithic model | Each domain has its own supersphere with its own vocabulary and kets. No shared parameters across domains (unless explicitly added). |
| Lossless output is NOT required | ShardFit's core engine (Section 1 of brief) | ShardFit requires output to match a reference implementation exactly. Your supersphere router produces approximate outputs — acceptable for research, not for ShardFit's core claim. |

---

## 6. Honest Assessment

| Use Case | Verdict | Why |
|----------|---------|-----|
| **Domain classification** (code vs. poetry vs. math) | ✅ Plausible | Cosine similarity between context and domain spheres should separate obvious domains well |
| **Text generation within a domain** | ⚠️ Limited | Supersphere is essentially weighted word2vec + softmax. It won't produce coherent long-form text, multi-step reasoning, or executable code. |
| **Running on Intel Celeron CPU** | ✅ Yes | A tiny supersphere model (vocab ~10K, dim ~128) is megabytes, not gigabytes. Will run on constrained hardware. |
| **Beating a 70B LLM on complex tasks** | ❌ No | A small geometric model cannot match a transformer's reasoning capacity, regardless of routing. |
| **Replacing ShardFit's core streaming engine** | ❌ No | Different architecture entirely. ShardFit streams one model's layers through VRAM; your idea routes between different models. |

### Bottom Line

The supersphere router is a **valid research direction** for lightweight, domain-routed language inference on constrained hardware. However, it is **not** a replacement for ShardFit's core lossless streaming engine. It should remain a separate experimental line of inquiry (as it is on the `experimental/supersphere-router` branch), distinct from the production-focused `master` branch.

---

## References

1. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.
2. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *Journal of Machine Learning Research*, 3(Feb), 1137-1155.
3. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NIPS*.
4. Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global vectors for word representation. *EMNLP*.
5. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2016). Enriching word vectors with subword information. *ACL*.
6. Gehring, J., Auli, M., Grangier, D., Dauphin, Y., & Dauphin, Y. (2017). Convolutional sequence to sequence learning. *ICML*.
7. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv:1701.06538*.
8. Lewis, M., Bhosale, S., Timokhin, T., Goyal, S., Mohamed, H., et al. (2021). RETRO: Improving language models by retrieving trillions of tokens. *arXiv:2112.04426*.
9. Leviathan, Y., Kalman, M., & Matias, Y. (2022). Fast inference from transformers via speculative decoding. *arXiv:2211.17192*.
10. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and sparsely activated routing. *arXiv:2101.03961*.
11. Chen, G., Borgeaud, S., Irving, G., et al. (2023). Speculative inference: Accelerating large language models via drafting and verification. *arXiv:2302.01318*.
12. Various authors (2023). Dynamic layer skipping / early exit (FiRST, D-LLM, DASH). *arXiv*.

---

*This document is part of the ShardFit project (https://github.com/Abhay27112006/Shardfit). It reflects the literature known as of November 2026.*
