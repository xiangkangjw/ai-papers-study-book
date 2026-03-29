# Chapter 7: Retrieval-Augmented Generation — Grounding Language Models in Knowledge

> Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

---

## 7.1 The Knowledge Problem in Language Models

A Transformer trained on a large corpus does not store facts in a lookup table. It stores them in billions of floating-point weights — compressed, distributed, and entangled with everything else the model learned. When you ask GPT-4 "Who was the first person to synthesize aspirin?", the answer is not retrieved from a database. It is reconstructed from gradients that adjusted weights during training whenever related text appeared in the corpus.

This distinction — **parametric memory** versus **non-parametric memory** — is the central tension that RAG addresses.

**Parametric memory** is what the model knows by virtue of its weights. Think of it as constants compiled into a binary: fast at inference, zero storage overhead, but immutable without recompilation (retraining). Facts baked in at training time stay frozen forever. The model cannot say "I don't know this yet" — it either hallucinates a plausible-sounding answer or silently produces the wrong one.

**Non-parametric memory** is an external datastore the model can query at runtime — a knowledge base, a document corpus, a SQL database. The model's parameters encode *how to use* the retrieved information, not *what* the information is. This is the database-lookup pattern: the schema (how to query, how to read results) is compiled in; the data is not.

The failure modes of pure parametric memory are severe in production systems:

1. **Hallucination**: The model generates text that is fluent and confident but factually wrong. There is no mechanism for the model to distinguish "I know this from training" from "I am pattern-matching to produce something plausible."

2. **Knowledge cutoff**: Training data has a hard boundary in time. A model trained through October 2023 cannot answer questions about events in 2024, no matter how capable it is otherwise.

3. **Stale knowledge without retraining**: Updating the model's knowledge requires a full retraining run or at minimum fine-tuning — expensive, slow, and operationally complex.

4. **No provenance**: Parametric models cannot cite sources because the knowledge is not stored with attribution. The model cannot say "this fact comes from document X."

5. **Capacity limits**: A model's effective knowledge capacity is bounded by parameter count. A 7B-parameter model cannot memorize the entirety of Wikipedia with high fidelity.

RAG solves all five problems simultaneously by separating the *knowledge* from the *reasoning*: the LLM learns how to read and synthesize; a retrieval system provides what to read.

---

## 7.2 The RAG Architecture

The Lewis et al. (2020) paper introduces RAG as a **probabilistic model** with two trainable components: a retriever and a generator. The formal objective is clean:

$$P(y \mid x) = \sum_{z} P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$$

where:
- $x$ is the input query
- $y$ is the output sequence
- $z$ is a retrieved document (or passage)
- $\eta$ are the retriever parameters
- $\theta$ are the generator parameters
- The sum marginalizes over the top-$k$ retrieved documents

The model does not pick a single document and condition on it. It treats retrieved documents as **latent variables** — unobserved intermediate states — and marginalizes over them. This is important: the model assigns probability mass to multiple possible retrieval paths and sums them. In practice, the sum is over the top-$k$ documents only (typically $k = 5$ or $k = 10$), making the marginalization tractable.

### Architecture Overview

```
          INPUT QUERY (x)
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────┐         ┌──────────┐
│  Query  │         │  Query   │
│ Encoder │         │ Encoder  │  (same encoder, different role)
│  (BERT) │         │  (BERT)  │
└────┬────┘         └────┬─────┘
     │                   │
     │ q(x)              │ (used at index build time)
     │                   ▼
     │           ┌───────────────┐
     │           │  Document     │
     │           │  Embeddings   │
     │           │  (FAISS index)│
     │           └───────┬───────┘
     │                   │
     └──── MIPS ─────────┘
               │
          top-k docs z₁...zₖ
               │
    ┌──────────┴──────────┐
    │   For each zᵢ:      │
    │  concat(x, zᵢ) ──► │
    │   BART Generator    │
    │   P(y | x, zᵢ)      │
    └──────────┬──────────┘
               │
    marginalize: Σᵢ P(zᵢ|x) · P(y|x,zᵢ)
               │
               ▼
          OUTPUT (y)
```

### The Retriever: Dense Passage Retrieval (DPR)

The retriever is built on **Dense Passage Retrieval** (Karpukhin et al., 2020). DPR is a **bi-encoder**: two separate BERT-based encoders that project queries and documents into the same dense vector space.

$$E_Q(x) \in \mathbb{R}^d$$

$$E_D(z) \in \mathbb{R}^d$$

$$s(x, z) = E_Q(x)^\top \cdot E_D(z)$$

The two encoders are independent — no cross-attention between query and document at retrieval time. This is architecturally deliberate: if the document encoder needs to see the query to produce a useful representation, you cannot pre-compute document embeddings offline. The bi-encoder decouples *indexing* (run once per document) from *querying* (run once per query at inference).

At indexing time, every document in the knowledge corpus is passed through $E_D$, producing a vector in $\mathbb{R}^d$. These vectors are stored in a FAISS index. At query time, the query is encoded with $E_Q$, and FAISS performs **Maximum Inner Product Search (MIPS)** to retrieve the top-$k$ documents by dot product similarity.

DPR was trained on Natural Questions using in-batch negatives plus hard negatives (documents retrieved by BM25 that are not answers). The training loss is softmax cross-entropy over the positive passage and all negatives: for a query $x$ with positive document $z^+$ and negatives $z^-_1, \ldots, z^-_n$,

$$\mathcal{L} = -\log \frac{e^{s(x, z^+)}}{e^{s(x, z^+)} + \sum_{i=1}^{n} e^{s(x, z_i^-)}}$$

This is a contrastive loss — the encoder learns a geometry where "query about aspirin synthesis" is close to "document describing aspirin synthesis" and far from irrelevant documents.

### The Generator: BART

The generator in the original RAG paper is **BART** (Lewis et al., 2019). BART is a denoising autoencoder: pre-trained by corrupting text and learning to reconstruct it, which makes it well-suited for conditional text generation. The generator takes a concatenation of the query and a retrieved document as input and produces the output sequence.

In modern RAG systems, the generator can be any sufficiently capable language model. The architecture works with GPT-style decoder-only models equally well — the retrieved documents are simply prepended as context.

### RAG-Sequence vs. RAG-Token

The paper introduces two variants that differ in how retrieved documents are used during generation:

**RAG-Sequence**: One set of top-$k$ documents is retrieved once per query. Every token in the output sequence is generated conditioned on the same retrieved documents.

$$P_{\text{RAG-Seq}}(y \mid x) = \sum_{z} P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$$

where $y = y_1 y_2 \cdots y_n$ (all tokens conditioned on the same $z$).

This is simpler. Retrieve once, generate the entire output. Marginalization is over whole output sequences given each document.

**RAG-Token**: A new retrieval can be performed at each output token position. Different tokens in the output sequence can be supported by different documents.

$$P_{\text{RAG-Token}}(y \mid x) = \prod_{t} \sum_{z} P_\eta(z \mid x) \cdot P_\theta(y_t \mid x, z, y_1 \cdots y_{t-1})$$

RAG-Token is more flexible — the model can pull from different sources as it generates — but more expensive: $k$ forward passes per token (in practice, approximated). RAG-Sequence tends to work better for tasks requiring coherent long-form generation; RAG-Token can better handle multi-fact answers where different facts live in different documents.

Empirically, RAG-Sequence outperforms RAG-Token on most benchmarks in the original paper.

---

## 7.3 The Retrieval Mechanism in Depth

### Dense vs. Sparse Retrieval

Before DPR, the dominant retrieval paradigm was **sparse retrieval** — primarily BM25, a probabilistic term-frequency model descended from TF-IDF.

BM25 scores a document $d$ for query $q$ as:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $f(t,d)$ is term frequency in the document, $|d|$ is document length, $\text{avgdl}$ is average document length, and $k_1$, $b$ are tuning parameters.

The key property: BM25 operates on an **inverted index** — a mapping from terms to document lists. It is extremely fast and requires no neural inference. It is, in the computer science sense, exact lexical matching with heuristic weighting.

The failure mode: **vocabulary mismatch**. A query asking about "myocardial infarction" will not retrieve documents that use "heart attack" unless the index contains synonym expansion. Sparse retrieval is fundamentally keyword matching; it has no notion of semantic equivalence.

Dense retrieval addresses this. When $E_Q$ and $E_D$ are trained on in-distribution data, semantically similar query-document pairs end up near each other in $\mathbb{R}^d$ regardless of surface-form word overlap. The embedding space captures meaning, not spelling.

The tradeoff:

| Property | BM25 (Sparse) | DPR (Dense) |
|---|---|---|
| Index type | Inverted index | Vector index |
| Inference at query time | Lookup + score | BERT forward pass |
| Out-of-vocabulary | Fails | Handles well |
| Domain transfer | Good (no training needed) | Requires in-domain data |
| Interpretability | Term weights | Black box |
| Memory | $O(\text{vocab} \times \text{docs})$ | $O(d \times \text{docs})$ |

Hybrid retrieval — BM25 for recall, dense reranking for precision — often outperforms either alone and is standard practice in production systems.

### FAISS and Approximate Nearest Neighbor Search

Once documents are embedded, retrieval is a nearest-neighbor problem in $\mathbb{R}^d$. Exact nearest-neighbor search over millions of vectors is $O(n \cdot d)$ per query — feasible for small corpora but impractical for Wikipedia (21 million passages in the DPR paper).

FAISS (Faiss, Johnson et al., 2017) solves this with **approximate nearest neighbor (ANN)** search. The two main index types:

**IVF (Inverted File Index)**: Cluster the dataset into $n_\text{list}$ Voronoi cells using k-means. At query time, search only the $n_\text{probe}$ nearest clusters. Reduces search to $O(n_\text{probe} \cdot n / n_\text{list} \cdot d)$. Trades recall for speed.

**HNSW (Hierarchical Navigable Small World)**: Build a multi-layer graph where short edges exist at the bottom (for local precision) and long edges at the top (for fast global navigation). Query traverses from the top layer down. $O(\log n)$ expected search time, high recall, but higher memory.

**PQ (Product Quantization)**: Decompose each vector into $m$ subvectors, quantize each subvector with a codebook. Reduces memory by 4–16×. Often combined with IVF as IVF+PQ.

For RAG at scale, FAISS with IVF+PQ is the standard. The engineering insight: retrieval quality depends heavily on ANN index configuration. Naively using default parameters can degrade recall by 10–20% versus exact search.

### The Asynchronous Index Problem

Here is a subtle engineering challenge the paper identifies: during joint training, the retriever's weights $\eta$ change at every gradient step. But the FAISS index contains embeddings computed from an earlier version of $\eta$. The index is stale.

One solution: refresh the entire index every $m$ steps. The Lewis et al. paper uses asynchronous refresh — the index is periodically rebuilt in the background while training continues with the stale index. This introduces a systematic bias (training with off-policy retrieval) but is practically necessary because re-encoding all of Wikipedia takes tens of minutes.

This is the RAG training analogue of the **target network** in DQN — you maintain a slowly-updated copy of the network to stabilize optimization.

### Code: Minimal RAG Pipeline

```python
# pip install sentence-transformers faiss-cpu

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Small document corpus — swap in your own documents to experiment
corpus = [
    "The Eiffel Tower is located in Paris, France, and was built in 1889.",
    "Python is a high-level programming language known for its readable syntax.",
    "The mitochondria is often called the powerhouse of the cell.",
    "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "Newton's first law states that an object in motion stays in motion.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
]

# Lightweight bi-encoder (~90MB on first run; no GPU needed)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Indexing phase (done once offline) ---
print("Encoding corpus...")
doc_embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
# Shape: [num_docs, embedding_dim], e.g. [8, 384]

embedding_dim = doc_embeddings.shape[1]
# IndexFlatIP = exact inner-product search; cosine sim after L2 normalization
index = faiss.IndexFlatIP(embedding_dim)
index.add(doc_embeddings)
print(f"Index built: {index.ntotal} documents, {embedding_dim}-dim embeddings\n")

# --- Query phase ---
def retrieve(query: str, top_k: int = 3) -> list:
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)
    return [(float(scores[0][i]), corpus[indices[0][i]]) for i in range(top_k)]

queries = [
    "Where is the Eiffel Tower?",
    "How do plants produce energy?",
    "What did Shakespeare write?",
]

for query in queries:
    print(f"Query: {query}")
    for rank, (score, doc) in enumerate(retrieve(query, top_k=2), 1):
        print(f"  [{rank}] score={score:.4f}  {doc}")
    print()
```

**What to observe:** The retriever finds semantically relevant documents without any keyword overlap — "How do plants produce energy?" retrieves the photosynthesis document because sentence embeddings capture meaning, not surface form. `all-MiniLM-L6-v2` produces 384-dimensional embeddings; production systems use larger models (e.g. `bge-large-en-v1.5`, 1024-dim) for higher recall. In a full RAG pipeline you would pass the top retrieved documents as context to an LLM generator alongside the original query.

---

## 7.4 Training RAG End-to-End

### Joint Optimization

RAG is trained end-to-end using standard maximum likelihood:

$$\mathcal{L}(\eta, \theta) = -\sum_{(x,y) \in \mathcal{D}} \log P(y \mid x) = -\sum_{(x,y) \in \mathcal{D}} \log \sum_{z} P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$$

The marginalization over $z$ makes this non-trivial. The log of a sum is not the sum of logs — you cannot push the gradient through trivially.

In practice, the sum over top-$k$ documents is computed:

$$\log P(y \mid x) \approx \log \sum_{i=1}^{k} P_\eta(z_i \mid x) \cdot P_\theta(y \mid x, z_i)$$

This requires running the generator $k$ times (once per document) and combining the results via a log-sum-exp:

$$\log \sum_{i} \exp\!\left(\log P_\eta(z_i \mid x) + \log P_\theta(y \mid x, z_i)\right)$$

Gradients flow through both $P_\eta$ (retriever softmax over top-$k$ scores) and $P_\theta$ (generator cross-entropy). The retriever is updated to increase the probability of documents that help the generator produce the correct output. This is implicit supervision: no explicit labels on which document is "correct."

### What the Retriever Learns

The implicit supervision is worth examining carefully. Suppose the correct answer to query $x$ is $y$. Documents that, when provided to the generator, increase $P_\theta(y \mid x, z)$ will receive higher gradient signal from the marginalization. Over training, the retriever learns to surface documents that are causally helpful for generation, not just documents that mention related terms.

This is an **indirect supervision signal**: the generator's feedback teaches the retriever what "useful" means through the end-to-end gradient, without explicit retrieval labels. The retriever does not need answer-span annotations; it only needs (query, target output) pairs.

### The REALM Connection

REALM (Guu et al., 2020), a concurrent paper, addresses the same problem with an important variant: it trains the retriever using the **ORQA** objective (Open-Retrieval Question Answering), propagating gradients directly through retrieval with exact MIPS using a periodically-refreshed index — essentially the same asynchronous refresh strategy. RAG adopts this approach. The convergence is theoretically questionable (optimization with stale indices), but empirically stable.

---

## 7.5 Results and Analysis

The Lewis et al. (2020) paper evaluates RAG on knowledge-intensive NLP tasks — tasks where the answer is unlikely to be producible from language modeling alone and requires specific world knowledge.

### Open-Domain QA

Three benchmarks: **Natural Questions** (real Google queries, Wikipedia answers), **TriviaQA** (trivia questions, multiple evidence documents), **WebQuestions** (questions about entities in Freebase).

RAG-Sequence on Natural Questions achieves 44.5% Exact Match, outperforming:
- T5-11B (closed-book): 34.5% EM — a 10+ point gap despite T5 having 11B parameters
- BM25 + BERT reader: ~26% EM
- REALM: 40.4% EM

The "closed-book" comparison is the most instructive. T5-11B must memorize facts in 11 billion parameters. RAG with a 400M-parameter BART and a DPR retriever over Wikipedia beats it by a substantial margin. Scaling parametric memory hits diminishing returns; external retrieval is a more efficient path to factual accuracy.

### Fact Verification: FEVER

FEVER (Fact Extraction and VERification) requires classifying claims as SUPPORTED, REFUTED, or NOT ENOUGH INFO using Wikipedia evidence. RAG achieves 71.8% label accuracy, competitive with specialized pipeline systems that use FEVER-specific architectures.

### Knowledge-Intensive Generation: Jeopardy Questions

Given an answer entity ("aspirin"), generate the corresponding Jeopardy question ("What is the brand name for acetylsalicylic acid?"). This requires knowing facts, not just retrieving them — the model must generate a grammatically correct question that is entailed by the fact.

RAG is evaluated on KG-to-text generation. The qualitative results show that retrieved documents meaningfully constrain generation: the model produces more specific, factually grounded output compared to BART without retrieval.

### Hallucination Reduction

A recurring finding: RAG models hallucinate less on factual questions. The mechanism is intuitive — the generator has access to a gold document that contains the answer, so it does not need to confabulate. The retriever can fail (retrieving irrelevant documents), but when it succeeds, the generator is strongly anchored to correct information.

---

## 7.6 The Modern RAG Ecosystem

The 2020 RAG paper established the architecture. The next four years produced a substantial engineering ecosystem around it. This section is organized as a system design problem.

### System Design View

```
                          ┌────────────────────────────────────────┐
                          │           RAG PIPELINE                 │
                          │                                        │
  User Query              │  ┌──────────┐    ┌──────────────────┐ │
──────────────► Preprocess►  │  Embed   ├───►│  Vector DB       │ │
                          │  │ (bi-enc) │    │  (FAISS/Pinecone)│ │
                          │  └──────────┘    └────────┬─────────┘ │
                          │                           │ top-k docs  │
                          │                  ┌────────▼─────────┐ │
                          │                  │   Reranker       │ │
                          │                  │  (cross-encoder) │ │
                          │                  └────────┬─────────┘ │
                          │                           │ top-n docs  │
                          │  ┌──────────────────────┐ │           │
                          │  │  LLM Generator       │◄┘           │
                          │  │  (GPT-4 / Claude /   │             │
                          │  │   Llama)             │             │
                          │  └──────────┬───────────┘             │
                          │             │                          │
                          └─────────────┼──────────────────────── ┘
                                        ▼
                                   Response + Citations
```

RAG is a **retrieval pipeline with an LLM as the final stage**. Every design decision in the pipeline affects end-to-end quality. The LLM generator is often the least tunable component; most engineering leverage is in the retrieval stack.

### Chunking Strategies

Documents must be split into passages before indexing. The chunk granularity directly affects retrieval quality.

**Fixed-size chunking**: Split on token count (e.g., 512 tokens with 50-token overlap). Simple, predictable, ignores document structure. Works well when documents are uniform in format.

**Semantic chunking**: Split at sentence or paragraph boundaries detected by NLP tools. Respects natural language units. More expensive to compute but better passage coherence.

**Recursive chunking**: Split by separators in priority order (`\n\n` → `\n` → `. ` → ` `), recursively subdividing until chunks are within size limits. This is the default in LangChain and works well for heterogeneous document types.

**Document-specific chunking**: PDFs have sections, code has functions, HTML has headers. Use document structure to determine splits. Highest quality but requires format-specific parsers.

Overlap between chunks (e.g., 10–15% overlap) ensures that answers near chunk boundaries are not missed. This is a standard engineering mitigation for the hard-split problem.

### Embedding Models

The embedding model is the single most impactful component after the LLM. Options:

**sentence-transformers** (Reimers & Gurevych, 2019): The open-source standard. Models like `all-mpnet-base-v2` (768-dim) and `multi-qa-mpnet-base-dot-v1` are trained specifically for retrieval tasks via contrastive learning. Free, fast, good quality.

**OpenAI `text-embedding-ada-002` / `text-embedding-3-large`**: API-based. `text-embedding-3-large` (3072-dim) is currently competitive with the best open-source models on MTEB. Advantage: no GPU required for embedding. Disadvantage: API cost, data leaves your infrastructure.

**Cohere Embed v3**: Strong multilingual support, explicit `input_type` parameter (distinguish query vs. document embeddings — important for bi-encoder performance). Competitive on MTEB retrieval benchmarks.

**BGE (BAAI General Embedding)**: Open-source models from Beijing Academy of AI. `bge-large-en-v1.5` ranks highly on MTEB. FlagEmbedding framework includes both bi-encoders and rerankers.

The **MTEB leaderboard** (Massive Text Embedding Benchmark, Muennighoff et al., 2022) is the standard evaluation for embedding models across retrieval, clustering, classification, and semantic similarity tasks. Consult it before choosing an embedding model.

### Vector Databases

| System | Type | Hosting | Best For |
|---|---|---|---|
| FAISS | Library | Self-hosted | Research, offline, GPU-accelerated ANN |
| Pinecone | Managed SaaS | Cloud | Production with no ops overhead |
| Weaviate | OSS + Cloud | Both | Hybrid (dense + sparse) search, GraphQL API |
| Chroma | OSS | Self-hosted | Local dev, embedded use |
| pgvector | Postgres extension | Self-hosted | Existing Postgres stack, SQL integration |
| Qdrant | OSS + Cloud | Both | Filtering + vector search, Rust-based |
| Milvus | OSS + Cloud | Both | Billion-scale, high-throughput |

For most production workloads: if you already have Postgres, try pgvector first (avoids operational complexity). For dedicated vector workloads at scale, Qdrant or Milvus. For hosted zero-ops, Pinecone. For research or offline processing, FAISS directly.

### Reranking: Two-Stage Retrieval

Bi-encoders are fast but approximate — they cannot capture fine-grained query-document interactions because query and document are encoded independently. **Cross-encoders** process the query and document *jointly*, allowing full cross-attention between them.

$$\text{Bi-encoder score:} \quad E_Q(x)^\top E_D(z) \quad O(1) \text{ per query after indexing}$$

$$\text{Cross-encoder score:} \quad \text{BERT}([\text{CLS}]\; x\; [\text{SEP}]\; z) \quad O(n \cdot d) \text{ per query, } n = \text{candidate count}$$

The two-stage pipeline:
1. **Recall stage**: Bi-encoder retrieves top-100 candidates (fast, approximate)
2. **Precision stage**: Cross-encoder reranks top-100 to top-5 (slow, accurate)

This is the standard industry pattern. Models like `cross-encoder/ms-marco-MiniLM-L-12-v2` (from sentence-transformers) are trained on MS MARCO passage ranking data and work well out-of-the-box. Cohere Rerank and Jina Reranker are hosted API alternatives.

The quality gain from reranking is substantial: on typical RAG benchmarks, two-stage retrieval improves faithfulness and answer correctness by 15–25% over single-stage bi-encoder retrieval.

---

## 7.7 Advanced RAG Patterns

### Self-RAG: Adaptive Retrieval (Asai et al., 2023)

A core inefficiency in vanilla RAG: retrieval is triggered for every query, whether or not retrieval is needed. A question like "What is 2+2?" does not benefit from Wikipedia lookup.

**Self-RAG** trains the LLM itself to emit special reflection tokens that control retrieval:

- `[Retrieve]`: Should retrieval be performed for this generation step?
- `[IsRel]`: Is the retrieved passage relevant?
- `[IsSup]`: Is the generation supported by the passage?
- `[IsUseful]`: Is this output useful?

The model is fine-tuned to predict these tokens interleaved with its output. At inference, the model self-determines when to retrieve, evaluates retrieved passages, and criticizes its own generations. This turns RAG from a fixed pipeline into a **metacognitive loop**.

Self-RAG outperforms RAG on ASQA (long-form QA), PopQA, and other benchmarks, while reducing unnecessary retrieval calls by 30–40%.

### RETRO: Retrieval-Integrated Pre-Training (Borgeaud et al., 2022)

RAG as described in Lewis et al. is a **post-training** addition — retrieval is bolted onto a pre-trained language model. RETRO (Retrieval-Enhanced Transformer) integrates retrieval into **pre-training** itself.

RETRO uses a chunked cross-attention mechanism: during pre-training, every chunk of the sequence retrieves its nearest neighbors from a 2 trillion-token datastore and attends to them. The architecture adds a frozen BERT retriever and a trainable chunked cross-attention layer to a standard Transformer decoder.

Key result: RETRO-3.5B (3.5B parameters + 2T-token retrieval) matches GPT-3 (175B parameters) on language modeling perplexity on the Pile benchmark. However, this comparison comes with caveats: the result holds specifically on the Pile, and the 2T-token retrieval datastore has significant overlap with the evaluation data. On tasks requiring generalization beyond the datastore, the gap narrows. Nonetheless, it demonstrates that retrieval can be a more parameter-efficient way to store knowledge than increasing model size.

This has profound implications: retrieval is not just an inference-time patch — it can replace parametric memory at scale.

### HyDE: Hypothetical Document Embeddings (Gao et al., 2022)

Bi-encoders are trained on (query, document) pairs, but the embedding spaces for queries and documents are not identical. A short natural language question is structurally different from a Wikipedia passage, which can create a distribution mismatch.

**HyDE** addresses this asymmetry: instead of embedding the query directly, use the LLM to generate a **hypothetical document** that would answer the query, then embed that hypothetical document.

```
Query:   "What causes aurora borealis?"
                    │
                    ▼
             LLM generates:
     "Aurora borealis is caused by charged particles
      from the solar wind interacting with Earth's
      magnetic field, exciting atmospheric gases..."
                    │
                    ▼
         Embed hypothetical document
                    │
                    ▼
     Search vector DB for similar real documents
```

The hypothetical document is in the same distribution as real documents — it is a dense, factual paragraph rather than a brief question. FAISS retrieves documents that are semantically similar to this hypothetical, which tend to be the actual relevant documents.

HyDE improves retrieval on low-resource domains where the retriever was not trained in-distribution. The cost: one LLM call per query just for retrieval.

### Multi-Hop Retrieval

Some questions require chaining evidence across multiple documents:

> "What country was the director of The Dark Knight born in?"

This requires: (1) retrieve who directed The Dark Knight (Christopher Nolan), then (2) retrieve where Christopher Nolan was born (UK). Single-pass retrieval cannot bridge this.

**Iterative retrieval** solves this with a loop:

```python
for step in range(max_hops):
    retrieve top-k documents for current query
    if answer found: break
    update query using retrieved information
    current_query = LLM.reformulate(original_query, retrieved_so_far)
```

Implementations include **IRCoT** (Trivedi et al., 2022), which interleaves retrieval with chain-of-thought reasoning, and **ReAct** (Yao et al., 2022), which frames retrieval as a tool in an agent action loop where the agent observes retrieved documents and takes actions to refine search.

### RAG vs. Fine-Tuning: When to Use Which

This is a common architectural decision point. The heuristics:

**Use RAG when**:
- Knowledge changes frequently (news, product catalogs, legal documents)
- You need source attribution and verifiability
- The knowledge corpus is large (millions of documents)
- You need to add knowledge without retraining
- Hallucination risk is high and correctness is critical

**Use fine-tuning (LoRA/PEFT) when**:
- You need to change the model's *behavior*, not its knowledge (tone, format, domain vocabulary)
- Latency is critical and you cannot afford retrieval overhead
- The knowledge is static and small enough to bake in
- You need strong performance on a narrow task distribution

**Use both when**:
- Domain-specific reasoning requires adapted behavior *and* access to a large knowledge base
- RAG with a fine-tuned retriever and a fine-tuned generator achieves highest performance on specialized domains

A common mistake: fine-tuning to inject facts. Fine-tuning on factual data reduces hallucination somewhat but is far less efficient than retrieval — it over-writes a small region of parameter space and degrades on other tasks. Use RAG for facts, fine-tuning for behavior.

---

## 7.8 Connections to Other Chapters

### RAG + LoRA/PEFT (Chapter 9)

LoRA fine-tuning (Chapter 9) adapts the generator's weights to a domain. Combined with RAG, you get a model that reasons in domain-specific language and retrieves from a domain-specific corpus. The two are complementary and additive. In practice: build a RAG system first, evaluate it, then apply LoRA fine-tuning to the generator if the base model's behavior is inadequate for the domain. Fine-tuning and retrieval optimize orthogonal axes — behavior and knowledge respectively.

### Multimodal RAG for VLMs (Chapter 10)

Vision-Language Models (Chapter 10) extend the RAG paradigm to multimodal retrieval. Instead of (or in addition to) a text document corpus, the retrieval index contains images, image-text pairs, or video frames. A multimodal query (an image plus a question) retrieves relevant visual context via CLIP-based embeddings (Radford et al., 2021), which map images and text into a shared embedding space via contrastive pre-training.

The architecture is identical to text RAG: replace the text bi-encoder with a multimodal encoder, replace the text corpus with a multimodal index. Applications include medical image QA (retrieve similar cases), document understanding (retrieve relevant diagrams), and product search (retrieve similar products).

### RAG as Tool Use in RL-Based Agents

From a reinforcement learning perspective, retrieval is an **action** in a partially observable environment. The agent (LLM) observes the query (partial state), selects a retrieval action (what to search for), receives an observation (retrieved documents), and produces an output. This framing makes RAG a natural component of ReAct and other LLM-as-agent architectures.

Self-RAG's reflection tokens are essentially a **value function approximation**: the `[Retrieve]` token estimates whether retrieval will increase the probability of a useful output. Training Self-RAG is analogous to training an agent with binary reward (retrieval helpful / not helpful). As RL-based reasoning systems (Chapter 12) scale, the decision of when and what to retrieve becomes a policy to be optimized — not a hard-coded pipeline step.

---

## 7.9 Summary

RAG decomposes the knowledge problem into two orthogonal concerns: *what to know* (the retrieval corpus, updatable at runtime) and *how to reason with retrieved information* (the generator, trained once). The key contributions of Lewis et al. (2020):

1. **Marginalization over retrieved documents** as latent variables — a principled probabilistic framing
2. **End-to-end joint training** of retriever and generator with implicit supervision
3. **Empirical demonstration** that retrieval outperforms much larger closed-book models on knowledge-intensive tasks

The architecture has proven remarkably durable. Four years after the paper, the core structure — bi-encoder retrieval, vector index, LLM generator — is the dominant pattern for enterprise AI systems. The variations (Self-RAG, RETRO, HyDE) refine specific weaknesses: adaptive retrieval, pre-training integration, query-document distribution mismatch.

For the systems-oriented reader: RAG is not just an NLP technique. It is a **system design pattern** — retrieval as infrastructure, LLM as query processor. The engineering challenges are database engineering challenges: index freshness, retrieval latency, chunk boundary effects, recall-precision tradeoff. Understanding those engineering dimensions is as important as understanding the training dynamics.

---

## Summary

- RAG (Retrieval-Augmented Generation) addresses the fundamental limitations of parametric memory --- hallucination, knowledge staleness, lack of provenance, and capacity constraints --- by separating *what to know* (an external retrieval corpus) from *how to reason* (the language model generator).
- The RAG architecture treats retrieved documents as latent variables and marginalizes over them: $P(y \mid x) = \sum_z P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$. End-to-end training propagates gradients through both the retriever and the generator, with the retriever learning to surface documents that are causally helpful for correct generation.
- Dense Passage Retrieval (DPR) uses a bi-encoder architecture with separate query and document encoders trained via contrastive loss. The bi-encoder design decouples indexing (run once per document) from querying (run once per query), enabling efficient retrieval over millions of passages using FAISS approximate nearest neighbor search.
- The practical RAG pipeline follows a two-stage retrieval pattern: a fast bi-encoder recall stage retrieves top-100 candidates, followed by a cross-encoder reranking stage that selects top-5 with full query-document cross-attention, improving faithfulness by 15--25% over single-stage retrieval.
- Self-RAG introduces metacognitive control through learned reflection tokens, enabling the model to decide when retrieval is needed and to evaluate the relevance and supportedness of retrieved passages, reducing unnecessary retrieval calls by 30--40%.
- RETRO integrates retrieval into pre-training itself via chunked cross-attention over a 2-trillion-token datastore, demonstrating that retrieval can substitute for parametric memory at scale: RETRO-3.5B matches GPT-3 175B on language modeling perplexity.
- RAG is best understood as a system design pattern --- retrieval as infrastructure, LLM as query processor --- where the engineering challenges (index freshness, chunking strategy, embedding model selection, recall-precision tradeoff) are as important as the training dynamics.

---

## Key Equations Reference

| Name | Equation | Section |
|---|---|---|
| RAG marginal probability | $P(y \mid x) = \sum_{z} P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$ | 7.2 |
| DPR similarity score | $s(x, z) = E_Q(x)^\top \cdot E_D(z)$ | 7.2 |
| DPR contrastive loss | $\mathcal{L} = -\log \frac{e^{s(x, z^+)}}{e^{s(x, z^+)} + \sum_i e^{s(x, z_i^-)}}$ | 7.2 |
| RAG-Sequence | $P_{\text{RAG-Seq}}(y \mid x) = \sum_z P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$ | 7.2 |
| RAG-Token | $P_{\text{RAG-Token}}(y \mid x) = \prod_t \sum_z P_\eta(z \mid x) \cdot P_\theta(y_t \mid x, z, y_{<t})$ | 7.2 |
| BM25 scoring | $\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d) + k_1(1 - b + b\frac{|d|}{\text{avgdl}})}$ | 7.3 |
| RAG training loss | $\mathcal{L} = -\sum_{(x,y)} \log \sum_z P_\eta(z \mid x) \cdot P_\theta(y \mid x, z)$ | 7.4 |
| MoCo momentum update | $\theta_k \leftarrow m \cdot \theta_k + (1-m) \cdot \theta_q$ | 7.3 |

---

## Exercises

**Exercise 7.1** *(Implementation)*

Build a minimal FAISS index from scratch using the `faiss` Python library. Embed 10,000 random 128-dimensional vectors representing "document embeddings." Implement two index variants: (a) a flat `IndexFlatIP` (exact inner product search) and (b) an `IndexIVFFlat` with $n_\text{list} = 100$ clusters and $n_\text{probe} = 10$. For 100 random query vectors, measure (i) mean Recall@10 of the IVF index relative to the flat index, and (ii) wall-clock query time for each. What recall do you observe, and how does it change as you vary $n_\text{probe}$ from 1 to 100? At what $n_\text{probe}$ does the IVF index achieve 95% recall relative to exact search?

**Exercise 7.2** *(Conceptual)*

You are building a RAG system for an internal engineering wiki containing 50,000 markdown documents. A colleague proposes chunking every document into fixed 512-token windows with no overlap. Identify at least three failure modes of this strategy for typical engineering wiki queries (e.g., "how do I configure the deployment pipeline?"), and for each failure mode propose a concrete mitigation. Then compare fixed-size chunking against recursive chunking and document-specific (structure-aware) chunking on two dimensions: implementation complexity and expected retrieval quality for this corpus. Which strategy would you recommend, and why?

**Exercise 7.3** *(Computation)*

A retrieval system returns the top-10 documents for each of 200 test queries. The ground truth specifies exactly one relevant document per query.

(a) You observe that the relevant document appears in the top-10 for 154 of 200 queries. Compute Recall@10.

(b) A cross-encoder reranker re-orders the top-10 to a top-3 shortlist; the relevant document appears in the shortlist for 138 queries. Compute Recall@3 after reranking.

(c) Write the general formula $\text{Recall}@k = m / N$ where $m$ is the number of queries with the relevant document in the top-$k$ and $N$ is the total query count. Prove or disprove: Recall@$k$ is always non-decreasing in $k$ for a fixed retrieval system.

(d) Why might Recall@3 after reranking be lower than Recall@10 before reranking, even though reranking is supposed to improve quality? What does this tell you about the reranker's role in the pipeline?

**Exercise 7.4** *(Conceptual)*

A startup is building a product that answers questions about a company's internal HR policies. The policies are updated quarterly. The technical lead proposes fine-tuning a 7B LLaMA model on the policy documents to "bake in" the knowledge. Write a structured comparison of this approach versus a RAG-based system, addressing: (a) knowledge staleness after each quarterly update and the cost to fix it, (b) attribution requirements ("which policy section says this?"), (c) one-time and ongoing infrastructure cost for each approach, and (d) one concrete scenario where fine-tuning would genuinely be the better choice. Conclude with a recommendation.

**Exercise 7.5** *(Computation)*

The DPR contrastive loss for a batch of $B = 32$ query-document pairs is:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s(x_i, z_i^+)}}{\sum_{j=1}^{B} e^{s(x_i, z_j)}}$$

where $z_j = z_j^+$ for the positive document of query $j$.

(a) Suppose all dot products $s(x_i, z_j) = 0$ except $s(x_i, z_i^+) = 2.0$ for all $i$. Compute the loss for a single query $i$ and for the full batch of 32.

(b) Now suppose the positive dot product increases to $s(x_i, z_i^+) = 5.0$ while all negatives remain at 0. Recompute the per-query loss. What does the change in loss tell you about the gradient magnitude as training progresses and the positives become well-separated?

(c) If the batch size doubles from $B = 32$ to $B = 64$, how does the loss change (assuming the same pattern of one positive at 2.0 and $B-1$ negatives at 0)? What is the implication for choosing batch size in DPR training?

**Exercise 7.6** *(Connection)*

Self-RAG trains a model to emit a `[Retrieve]` token to decide whether retrieval is needed at each generation step. Frame this as a reinforcement learning problem: define the state, action space, and a plausible reward function. How does this relate to the PPO objective in Chapter 10? Self-RAG is not actually trained with RL — it uses supervised fine-tuning on synthetic data generated by a teacher model. Explain how this SFT approach approximates the RL objective, and identify one systematic failure mode that pure SFT-based training would have compared to genuine RL training.

---

## References

- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Lewis, M. et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv:1910.13461.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
- Guu, K. et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*.
- Borgeaud, S. et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *ICML 2022*.
- Asai, A. et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.
- Gao, L. et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. *ACL 2023*.
- Johnson, J. et al. (2017). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- Trivedi, H. et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023*.
- Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Muennighoff, N. et al. (2022). MTEB: Massive Text Embedding Benchmark. *EACL 2023*.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
