# Related Papers

| Year | Paper | Authors | Core Idea |
|------|-------|---------|-----------|
| 1991 | Adaptive Mixtures of Local Experts | Jacobs, Jordan, Nowlan, Hinton | Mixture of Experts: a gating network routes inputs to specialized expert networks. |
| 2003 | A Neural Probabilistic Language Model | Bengio, Ducharme, Vincent, Jauvin | Words mapped to continuous vectors; next-word prediction using a feedforward network over embeddings. |
| 2013 | Distributed Representations of Words and Phrases (word2vec) | Mikolov, Chen, Corrado, Dean | Word vectors learned by predicting surrounding words; cosine similarity measures semantic similarity. |
| 2014 | GloVe: Global Vectors for Word Representation | Pennington, Socher, Manning | Word vectors trained so dot product approximates log co-occurrence probability. |
| 2016 | Enriching Word Vectors with Subword Information (FastText) | Bojanowski, Grave, Joulin, Mikolov | Words represented as bags of character n-grams for out-of-vocabulary word vectors. |
| 2017 | Convolutional Sequence to Sequence Learning | Gehring, Auli, Grangier, Dauphin | Convolutional architectures compete with Transformers for sequence modeling. |
| 2022 | Fast Inference from Transformers via Speculative Decoding | Leviathan, Kalman, Matias | Small draft model proposes tokens; large target model verifies them in parallel. |
| 2022 | Switch Transformers: Scaling to Trillion Parameter Models | Fedus, Zoph, Shazeer | Sparsely-gated MoE: top-1 expert routing per token at trillion-parameter scale. |

---

## Similarities with ShardFit

| Paper | Similarity |
|-------|-----------|
| Adaptive Mixtures of Local Experts (1991) | Both split a model into specialized parts and route computation. ShardFit routes shards through VRAM slots rather than routing tokens to experts. |
| A Neural Probabilistic Language Model (2003) | Both use neural language models to predict next tokens. ShardFit streams the same model weights through VRAM instead of keeping them all in memory. |
| Distributed Representations of Words and Phrases (word2vec, 2013) | Both work with model weights as vectors. ShardFit streams these weight vectors through VRAM in contiguous shard buffers. |
| GloVe (2014) | Both work with learned vector representations. ShardFit streams these learned weight matrices through VRAM slots. |
| Enriching Word Vectors with Subword Information (FastText, 2016) | Both decompose model weights into smaller sub-components. ShardFit decomposes model layers into shards that fit VRAM slots. |
| Convolutional Sequence to Sequence Learning (2017) | Both are non-Transformer architectures for sequence modeling. ShardFit works with standard Transformer weights but streams them instead of keeping them resident. |
| Fast Inference from Transformers via Speculative Decoding (2022) | Both use a smaller/faster component to accelerate inference. ShardFit's Phase 5 (speculative decoding with resident draft model) builds directly on this. |
| Switch Transformers (2022) | Both handle models too large for single-device memory. ShardFit streams all shards through VRAM instead of routing tokens to experts. |

---

## Differences from ShardFit

| Paper | Difference |
|-------|-----------|
| Adaptive Mixtures of Local Experts (1991) | ShardFit does **not** use routing or gating. Every shard runs on every pass; no expert selection. ShardFit is lossless; MoE routing is approximate. |
| A Neural Probabilistic Language Model (2003) | ShardFit does not change the model architecture. It streams the same weights through VRAM slots instead of keeping them all in memory at once. |
| Distributed Representations of Words and Phrases (word2vec, 2013) | ShardFit is not a word embedding method. It streams full Transformer model weights through VRAM to run models larger than GPU memory. |
| GloVe (2014) | ShardFit does not train word vectors. It loads and streams pre-trained model weights through VRAM shards for inference. |
| Enriching Word Vectors with Subword Information (FastText, 2016) | ShardFit does not use subword information. It splits Transformer layers into contiguous shards for VRAM streaming. |
| Convolutional Sequence to Sequence Learning (2017) | ShardFit works with standard Transformer architectures. It does not change the model architecture; it only changes how weights are loaded into VRAM. |
| Fast Inference from Transformers via Speculative Decoding (2022) | ShardFit's speculative decoding (Phase 5) uses a resident draft model verified by the streamed model. The core streaming mechanism is independent of speculative decoding. |
| Switch Transformers (2022) | ShardFit does **not** use expert routing. Every layer runs on every pass; no token routing to experts. ShardFit streams all weights through VRAM slots instead of selecting experts per token. |

