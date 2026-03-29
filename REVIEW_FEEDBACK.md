# Editorial Review: Understanding Deep Learning Textbook

**Reviewer:** Technical Editor
**Date:** 2026-03-29
**Audience reminder:** Experienced software engineers and CS graduates (non-AI degrees) with years of professional experience. They want to understand the foundational research papers behind what's being built. This is NOT for students, NOT for deployment/ops guidance — it's for building deep conceptual understanding of the science.

---

## Part 1: Content Gaps

The chapter arc (VAE > GAN > Transformer > GPT/BERT > ViT > RAG > Diffusion > LoRA > RL/VLM) tracks the major research lineages well. However, there are six topics that represent foundational research contributions missing from the book. None require a new chapter — they are each section-sized additions.

### GAP-1: Chain-of-Thought Reasoning [HIGH — new section, ~5-8 pages]

**Where:** Ch05 (after scaling/emergence discussion) or Ch10 (before test-time compute)

**What's missing:** CoT is treated as a prompting technique (~2 sentences in Ch10) rather than as a foundational research result. The key papers to cover:

- Chain-of-Thought Prompting (Wei et al., 2022) — the finding that intermediate reasoning steps unlock latent capabilities in large models
- Self-Consistency (Wang et al., 2023) — majority voting over sampled reasoning paths
- Tree of Thoughts (Yao et al., 2023) — structured search over reasoning branches
- Least-to-Most Prompting (Zhou et al., 2023) — decomposing complex problems into subproblems
- The theoretical angle: Feng et al. (2023) showed CoT lets Transformers solve problems outside their direct expressivity class — the format of the output changes what the model can compute

**Why it matters:** CoT is the direct intellectual predecessor to the test-time compute / reasoning line (o1, R1). Without understanding CoT as a research contribution, the reader cannot understand why DeepSeek-R1 trains a model to generate long reasoning traces with RL. These two topics form a natural narrative arc:

> CoT showed reasoning traces help > self-consistency showed sampling multiple helps > process reward models showed you can train verifiers > o1/R1 showed you can train the model to generate better traces with RL

This story should be told as a connected sequence.

### GAP-2: Test-Time Compute / Inference-Time Scaling [HIGH — expand existing, ~5-8 pages]

**Where:** Ch10 — build directly on the CoT section

**What's missing:** Ch10 mentions DeepSeek-R1 briefly but doesn't cover the conceptual framework:

- Why spending more compute at inference can substitute for model scale
- How verification enables search over reasoning traces
- Process reward models vs. outcome reward models (Lightman et al., 2023)
- Inference-time scaling laws (the counterpart to training-time scaling laws in Ch05)
- The o1/o3 paradigm and DeepSeek-R1's RL-driven chain-of-thought

This is the defining research direction of 2024-2025 and the "Frontiers" chapter should cover it prominently.

### GAP-3: Mixture of Experts [HIGH — new section, ~4-6 pages]

**Where:** Ch05 (after scaling laws discussion)

**What's missing:** MoE is a fundamental architectural idea about conditional computation, not an engineering trick. Key papers:

- Switch Transformer (Fedus et al., 2022)
- GShard (Lepikhin et al., 2021)
- Mixtral (Jiang et al., 2024)
- DeepSeek-V2/V3 MoE architecture

Cover: sparse gating, expert routing, load balancing, why MoE changes the relationship between parameter count and compute cost. A reader encountering "Mixtral 8x7B" should understand what "8x" means.

### GAP-4: State Space Models / Mamba [HIGH — new section, ~4-6 pages]

**Where:** Ch04 (as a counterpoint to attention) or Ch10 (alternative paradigms)

**What's missing:** The book's narrative is "RNN > Transformer > everything." SSMs challenge that:

- S4 (Gu et al., 2022)
- Mamba (Gu & Dao, 2023)
- Mamba-2 and hybrid architectures (Jamba, etc.)

Cover: why the field circled back to recurrence, the O(n) scaling advantage, no KV cache requirement, long-range modeling. Position this as the main alternative paradigm — the reader should understand the attention-vs-recurrence tradeoff space.

### GAP-5: Contrastive Learning as a Paradigm [MEDIUM — new section, ~3-4 pages]

**Where:** Ch01 (foundations) or Ch06 (before CLIP)

**What's missing:** CLIP (Ch06), DPR (Ch07), and DINO (Ch06) all use contrastive losses, but the paradigm itself is never introduced. Key papers:

- SimCLR (Chen et al., 2020)
- MoCo (He et al., 2020)
- InfoNCE loss

Establish "learning by comparing positives and negatives" as a foundational idea so readers recognize the pattern when it recurs across chapters.

### GAP-6: Tokenization / BPE [MEDIUM — new section, ~2-3 pages]

**Where:** Ch04 or Ch05

**What's missing:** BPE (Sennrich et al., 2016) is a foundational paper. WordPiece gets a paragraph in Ch05 for BERT, but BPE for GPT is never discussed. Tokenization choices have implications that appear in many subsequent papers (multilingual performance, arithmetic failures, byte-level models). Cover the "how does text become input?" question that underpins every NLP paper in the book.

### GAP-7: JEPA [LOW-MEDIUM — new subsection, ~1-2 pages]

**Where:** Ch06 (after MAE/DINO) or Ch10 (open problems)

**What's missing:** JEPA (LeCun, 2022 position paper), I-JEPA (Assran et al., 2023), V-JEPA (Bardes et al., 2024). Frame as "an alternative paradigm that predicts in representation space rather than pixel space." It hasn't had the field-reshaping impact of Transformers or diffusion yet, but it represents a fundamentally different bet on representation learning. A page or two is sufficient — the reader should know what JEPA is if they encounter it, but it doesn't need the same weight as core chapters.

---

## Part 2: Factual Errors (Must Fix)

These are errors that will undermine credibility with technically rigorous readers.

### ERRATA-1: Wrong chapter cross-references in Ch01
Ch01 Section 1.5.2 and Section 1.7 say "Transformer architecture (Chapter 2)" — should be **Chapter 4**. The closing line "Turn the page. We start with the architecture that changed everything" also implies the next chapter is the Transformer, but it's actually VAE (Ch02).

### ERRATA-2: Wrong gradient analysis in Ch03 Section 3.3.2
"the gradient of log(1-y) with respect to y is -1/(1-y), which approaches 0 as y -> 0." The value -1/(1-y) approaches **-1** as y -> 0, not 0. The vanishing gradient problem is real but the explanation is incorrect. The paragraph conflates the gradient of the loss w.r.t. the discriminator output with the gradient w.r.t. the generator's parameters through the chain rule.

### ERRATA-3: Wrong DPR loss formula in Ch07 Section 7.3
The formula shows binary cross-entropy, but DPR actually uses softmax over the positive and all negatives:
```
-log( exp(s(x, z+)) / (exp(s(x, z+)) + sum_i exp(s(x, z-_i))) )
```
This distinction matters — the softmax formulation is what makes in-batch negatives work.

### ERRATA-4: Wrong compression math in Ch02 Section 7.1 (VQ-VAE context)
"512x512 pixel space is ~65,000 dimensions" — a 512x512 RGB image is 786,432 dimensions. "64x64 latent space is ~4,000 dimensions" — it's 16,384 (64x64x4). The "16x fewer" claim is wrong (should be ~48x).

### ERRATA-5: Left/right architecture error in Ch05 Section 5.2.1
"the left half of the architecture from Vaswani et al. (2017)" — GPT uses the **right** half (decoder side), not the left half (encoder side).

### ERRATA-6: Memory calculation in Ch09 Section 9.1
Claims 12 bytes per parameter, but omits the fp32 master copy required for mixed-precision training. Actual cost: 2 (fp16 weights) + 2 (gradients) + 4 (fp32 master) + 4 (m1) + 4 (m2) = **16 bytes**, not 12. The 84 GB figure underestimates by ~25%. Also: "An A100 80GB GPU can barely hold it" — 84 GB exceeds 80 GB, so it **cannot** hold it. That's actually a stronger statement; use it.

### ERRATA-7: H100 TFLOPS in Ch01
"roughly 1,000 TFLOPS at FP8" — H100 SXM5 is ~1,979 TFLOPS dense FP8 (~3,958 with sparsity). Either cite the specific SKU or use "roughly 2,000 TFLOPS."

### ERRATA-8: Swin Transformer shift in Ch06 Section 6.6.2
Uses P/2 (patch size) for the window shift amount — should be floor(M/2) where M is the window size. P refers to patch size elsewhere in the chapter.

### ERRATA-9: Exercise 5.3 notation collision
Uses N* for both tokens and parameters. Should be D* for tokens and N* for parameters, matching the scaling law notation from Section 5.5.3.

### ERRATA-10: Label smoothing approximation in Ch04 Section 4.6.2
"epsilon_ls / V ~ 0.0001" assumes V ~ 1000. For typical NLP vocabularies (30K-50K), this is ~0.000002-0.000003. Off by two orders of magnitude.

### ERRATA-11: ViT accuracy attribution in Ch06 Section 6.2
"ViT-Large reaches 87.76%" — the original paper reports 87.76% for ViT-H/14, not ViT-L/16. Check Table 2 of Dosovitskiy et al.

### ERRATA-12: Frequency vs. wavelength in Ch04 Section 4.5.2
Describes "2pi x 10000" as a frequency when it is actually the wavelength (lowest frequency). The description inverts the relationship.

### ERRATA-13: InstructGPT claim in Ch10 Section 10.2
"InstructGPT model... despite having only 1.3B parameters" implies InstructGPT was only 1.3B. It was trained at 1.3B, 6B, and 175B. The 1.3B variant was preferred over vanilla GPT-3 175B, but the flagship results used the 175B model. Clarify.

### ERRATA-14: Bernoulli notation in Ch02 Section 2.4.1
"a Bernoulli with probabilities p_theta(z)" — should be p_theta(x|z). As written, p_theta(z) looks like the prior over z.

### ERRATA-15: Progressive growing attribution in Ch03 Section 3.5.5
Listed as a StyleGAN innovation. Progressive growing was introduced in ProGAN (Karras et al., 2018). StyleGAN used it but did not introduce it.

---

## Part 3: Technical Precision Issues (Should Fix)

### Ch01
- TP-1: Universal Approximation Theorem conflates Cybenko (1989, sigmoidal only) and Hornik (1991, general). Distinguish their contributions.
- TP-2: "Backward pass is roughly 2x the forward pass" — more accurately 2-3x; for attention, can be significantly more. Add caveat.
- TP-3: Adam bias correction exponents beta_1^t, beta_2^t — never states that t is the iteration count (exponentiated, not multiplied).
- TP-4: Dropout "ensemble of 2^n sub-networks" interpretation is debated (Gal & Ghahramani 2016 Bayesian interpretation). Add brief note.
- TP-5: GELU definition gives x * Phi(x) but doesn't specify Phi is the standard normal CDF N(0,1).
- TP-6: Frobenius norm mentioned but never defined. Define it since it appears in LoRA analysis (Ch09).

### Ch02
- TP-7: PCA description says "maximizes reconstructed variance" — nonstandard. PCA minimizes reconstruction error / maximizes projection variance.
- TP-8: ELBO sign ambiguity — the text calls it "the loss" but the formula is a quantity to maximize (the loss is the negative ELBO). Clarify.
- TP-9: "Alternative decomposition" block in 2.4.5 is identical to the derivation two lines above. Either cut or make genuinely different.

### Ch03
- TP-10: VAE comparison "VAEs minimize KL(p_data || p_g) (approximately)" — imprecise. VAEs maximize the ELBO, which is a lower bound on log-likelihood. Make precise.
- TP-11: Pix2Pix loss (3.5.4) overloads y (target image) with y from 3.5.3 (conditioning label).
- TP-12: Orbit claim in 3.4.2 — the discrete simultaneous update system actually diverges (spirals outward), not stable orbits. This is actually a stronger argument for the instability point being made.
- TP-13: JSD formula — state which log base is assumed.

### Ch04
- TP-14: Pseudocode in 4.4.5 implements post-norm but text says pre-norm is standard. Label the pseudocode as "original post-norm formulation" or provide both variants.
- TP-15: FlashAttention — "reduces HBM reads from O(n^2) to O(n^2 d/M)" looks like it gets worse. Add clarifying sentence that M (SRAM size) makes d/M < 1.
- TP-16: "DALL-E 2" grouped with U-Net cross-attention — better describes Stable Diffusion. DALL-E 2 uses CLIP + diffusion prior + unCLIP decoder.

### Ch05
- TP-17: "bidirectional context makes this harder than autoregressive prediction" — debatable. MLM has more information per prediction. Qualify.
- TP-18: Scaling law exponents are from Kaplan et al. (2020) but Chinchilla (Hoffmann et al., 2022) found different exponents. Clarify which paper's exponents are cited.
- TP-19: "Emergent abilities" — Schaeffer et al. (2023) argued emergence is an artifact of nonlinear metrics. Give this counterargument more weight.
- TP-20: PPO objective shown is the reward-with-KL-penalty formulation, not the PPO clipped surrogate. Clarify which variant is described.
- TP-21: AlexNet cited for transfer learning — the transfer result is from Donahue et al. (2014) / Razavian et al. (2014).

### Ch06
- TP-22: "Superhuman performance on ImageNet" attributed to ResNet — the human error rate measurement conditions are debatable. Add caveat.
- TP-23: LLaVA "uses 336x336" — that's LLaVA 1.5, not 1.0 (which uses 224x224). Specify version.
- TP-24: Dropout "only during fine-tuning" — depends on pre-training dataset size. Not always true.

### Ch07
- TP-25: BART and RAG framed as "same year" — BART is Lewis et al. (2019), RAG is Lewis et al. (2020). Different papers, different years.
- TP-26: "RETRO-3.5B matches GPT-3" — only on Pile perplexity with overlapping datastore. Overstated.
- TP-27: "Knowledge distillation in the broad sense" for RAG's training signal — this term has a specific meaning (Hinton et al., 2015). Misleading here.
- TP-28: FAISS provides approximate nearest-neighbor, not exact. Emphasize the recall-accuracy tradeoff.

### Ch08
- TP-29: "DDPM requires every intermediate step to maintain the Markov property" — DDPM can skip steps too (with degraded quality). DDIM enables skipping without variance accumulation.
- TP-30: "Computational cost scales quadratically with resolution" — true for attention layers but not convolutions. Attention is only applied at lower resolutions in pixel-space models. Overstated.
- TP-31: CLIP model version ambiguous — SD 1.x uses CLIP ViT-L/14 (768-dim), SD 2.x uses OpenCLIP ViT-H/14 (1024-dim). Specify.
- TP-32: Probability flow ODE uses undefined f(t) and g(t). Either define or describe qualitatively.

### Ch09
- TP-33: QLoRA merging "impossible" — recent libraries support approximate merging. Soften to "not straightforward."
- TP-34: LoRA+ ratio "lambda ~ 16" — the paper's ratio varies by model/task. Say "typically 4-16."
- TP-35: "Diminishing returns by r=64" — task-dependent. Some tasks (math reasoning) benefit from r=128+.
- TP-36: Performance table ranges attributed to "NLP benchmarks" — note they may differ for vision/multimodal.

### Ch10
- TP-37: PPO clipping for negative advantages — the explanation is imprecise. Clarify the behavioral consequence: "prevents reducing the action's probability too aggressively."
- TP-38: DPO limitations understated. Missing: distribution shift (offline), overfitting to preference data. Mention at minimum the offline/distribution-shift issue.
- TP-39: GRPO attributed to Shao et al. (2024) but not connected to DeepSeek-R1 (Section 10.3). Link them.
- TP-40: Flamingo "Perceiver Resampler layers" conflates two components. The Perceiver Resampler is one module; gated cross-attention layers are separate.
- TP-41: RT-2 — "treating motor commands as text tokens" — they are specially defined action tokens in the vocabulary, not literally text.
- TP-42: CLIP "matched ResNet-50" — only with the largest model (ViT-L/14@336px). Specify.

---

## Part 4: Structural & Consistency Issues

### S-1: No exercises in any chapter except Ch05
Every chapter needs exercises. For this audience, prioritize:
- Derivation exercises ("verify the closed-form KL," "show the ELBO gap equals KL to the true posterior")
- Computation exercises ("compute one attention head's output for this input")
- Paper-reading exercises ("read Section 3 of [paper], identify the assumption that enables...")
- Conceptual exercises ("why can't you just maximize log-likelihood directly for VAEs?")

### S-2: Inconsistent math formatting across chapters
- Ch06: LaTeX throughout
- Ch07: code blocks for formulas
- Ch09: inline code notation
- Standardize to LaTeX everywhere. This is non-negotiable for a textbook.

### S-3: Inconsistent end-of-chapter apparatus
- Ch03 has "Key Equations Reference" table — add to all chapters
- Ch02 has "References," Ch03 has "Further Reading" — pick one format
- Some chapters have "Connections Forward," others don't — standardize

### S-4: Inconsistent citation format
- Some use arXiv year, some use venue year
- Some use (Author, Year), others use Author et al. (Year)
- Constitutional AI, RLAIF mentioned without citations in Ch05 while DPO gets one
- Pick one convention and apply consistently

### S-5: Inconsistent boldface for vectors/matrices
- Ch01: LSTM gate vectors (f_t, i_t, o_t) not bolded despite being vectors
- Ch04: W^O sometimes bolded, sometimes not
- Ch06: mixed LaTeX and inline code for the same quantities
- Define a notation convention and enforce it

### S-6: Missing notation summary table
Ch01 establishes notation but there is no reference table. Add one at the end of Ch01 (or as an appendix) listing all symbols used throughout the book.

### S-7: Missing "How to Read This Book" in frontmatter
This audience is time-constrained. Add:
- Chapter dependency diagram (can I skip to Ch09 directly?)
- Self-assessment checklist in Ch01 ("if you can answer these 5 questions, skim this chapter")
- Minimal paths for specific interests (e.g., "LLM path: Ch01 > Ch04 > Ch05 > Ch07 > Ch09")

### S-8: KL divergence defined too late
Used extensively in Ch02 (VAE) but not formally defined until Ch03. Define it in Ch01 (information theory section) or at first use in Ch02.

### S-9: Ch06 and Ch10 VLM overlap
Ch06 Section 6.7 covers CLIP, LLaVA, and Flamingo in detail (InfoNCE loss, LLaVA shape flow, Perceiver Resampler). Ch10 also covers these. Decide which chapter is canonical for VLMs and make the other a brief cross-reference.

### S-10: No figure/diagram references anywhere
A textbook on visual/generative models without figures is a significant weakness. Even if figures are produced separately, add placeholder references throughout: "[See Figure 2.1: VAE encoder-decoder architecture]", "[See Figure 4.3: Multi-head attention]", etc.

### S-11: Missing runnable code in most chapters
Chapters with zero code: Ch02, Ch03, Ch06, Ch07, Ch08, Ch10. For this audience of experienced SWEs, runnable code is how they verify understanding. Priority additions:
- Ch02: Minimal PyTorch VAE on MNIST (~40 lines)
- Ch03: WGAN-GP training loop
- Ch07: RAG pipeline with sentence-transformers + FAISS (~30 lines)
- Ch08: `diffusers` pipeline inference example
- Ch10: CLIP zero-shot classification (~10 lines)

Where pseudocode exists, clarify whether it's runnable Python or pseudocode. Ch01's dropout code uses `random(h.shape)` which is not valid Python. Ch04's pseudocode mixes `nn.Module` patterns with undefined functions.

---

## Part 5: Audience-Specific Feedback

Given the target audience (experienced SWEs/CS grads, non-AI background), these points override some standard textbook advice:

### What's working well for this audience — keep it
- The direct "you" tone ("You have spent a career building systems") is appropriate and engaging. Do not make it more academic.
- Engineering analogies (SQL JOIN for attention, "constants compiled into a binary" for parametric knowledge, Unix pipes for composition) are strengths. Polish them but don't remove them.
- Ch09's opening cost analysis ("a 7B model costs 84 GB to fine-tune") is perfect pedagogy — state the practical problem, then introduce the solution. Consider this as a model for how every chapter opens.
- Terms like "embarrassingly parallel" need no explanation for this audience.

### What needs adjustment for this audience
- **ML-specific concepts need more scaffolding.** This is the core gap. CS/math foundations (linear algebra, calculus, probability, data structures) can be assumed. ML-specific concepts cannot:
  - KL divergence, ELBO, variational inference (Ch02) — these readers likely haven't seen these since an optional information theory elective, if ever
  - Reinforcement learning (Ch10) — MDPs through PPO in ~2 pages is too compressed. Expand significantly or add a prerequisite note.
  - Mean-field approximation (Ch02) — needs context connecting to variational inference literature
  - Bayesian inference basics — never introduced but underpin the VAE chapter. Add a short primer.

- **Add worked numerical examples.** This audience learns by computing, not trusting derivations. Priority:
  - One SGD step with actual numbers (Ch01)
  - One attention score computation (Ch04)
  - One ELBO term evaluation (Ch02)
  - One diffusion forward/reverse step (Ch08)

- **Add self-assessment checkpoints in Ch01.** "If you can answer these 5 questions, skim this chapter" at the start. "Review" vs. "New Concepts" labels on sections so readers can self-navigate.

- **Section 1.6 (Training Infrastructure) is high-value for this audience, not misplaced.** SWEs think in terms of GPUs, memory, and distributed systems. Keep it.

- **Ch07 Sections 7.6-7.7 (practitioner guide) are the most valuable part for this audience.** The tonal shift from paper analysis to engineering guidance is fine — consider framing explicitly: "Here's the science > here's how to build it."

- **Remove "(relevant to your RL background)" in Ch07 Section 7.7** — breaks the fourth wall, reads as a note to a specific person rather than textbook prose.

---

## Part 6: Per-Chapter Copy Editing Notes

### Ch01
- CE-1: Section 1.1 — "Machine learning inverts this: you specify..." — colon introduces a list where the subject shifts. Rephrase.
- CE-2: "a fact worth internalizing" (Section 1.2.3) — editorial voice feels out of place in technical body. Rephrase to "a connection we will revisit in Chapters 2 and 9."
- CE-3: Inconsistent citation style — some (Author, Year), others Author et al. (Year). Standardize.
- CE-4: CNN convolution formula uses delta-i, delta-j as kernel offsets — unusual and risks confusion with Kronecker delta and backprop delta used nearby. Use m, n or k_1, k_2.

### Ch02
- CE-5: Epigraph quotes Kingma & Welling but is a paraphrase, not a direct quote. Either find the real quote or drop quotation marks.
- CE-6: Section 2.5.6 — long sentence about Bowman et al. buries the key finding. Split into two sentences.
- CE-7: "domains you likely encounter" — fine for this audience, keep as is.

### Ch03
- CE-8: Inconsistent contractions — "Let's" (3.2.1), "There's" (3.3.2), "can't" mixed with formal prose. Pick one register.
- CE-9: "WGAN-GP became a go-to baseline" — "go-to" is colloquial. Use "widely adopted baseline."
- CE-10: "the situation is even worse" and "Balance is critical and fragile" — editorial commentary without evidence. Back with specific references.
- CE-11: Further Reading section — inconsistent format (some entries have paper titles in quotes, others don't).

### Ch04
- CE-12: "catastrophic underutilization of hardware" — "catastrophic" is hyperbolic. Use "severe."
- CE-13: "doubling GPUs approximately halved training time" — only with perfect data-parallel scaling. Add "in the ideal case."
- CE-14: Missing __init__ in EncoderLayer pseudocode — if this is pseudocode, don't use real Python list comprehensions in TransformerEncoder.__init__. Pick one abstraction level.

### Ch05
- CE-15: BERT-Base 110M vs. GPT 117M with identical config (12 layers, 768, 12 heads) — explicitly explain why counts differ (segment embeddings, vocabulary size).
- CE-16: "cross-entropy is computed against the original token" — say "the original (pre-masking) token."
- CE-17: "The technical innovations were incremental relative to InstructGPT" — ChatGPT details never fully published. Soften to "are believed to be incremental."
- CE-18: Constitutional AI (Bai et al., 2022) mentioned without citation while DPO gets one.

### Ch06
- CE-19: "depth and small 3x3 kernels were the key variable" — "key variable" awkward; also it's two variables. Use "key design principles."
- CE-20: "a bet on data scale over architectural prior" — "priors" (plural).
- CE-21: "This would not work if images were processed by CNNs" — overstated. CNN features can be reshaped into sequences. Say "less natural" rather than "would not work."
- CE-22: Chapter subtitle "When Transformers Replaced Convolutions" — ConvNeXt showed modernized CNNs match ViT. Consider softening.
- CE-23: "the ResNet team at Facebook AI" — insider commentary, not textbook prose. Remove.

### Ch07
- CE-24: "Transformer" lowercase in 7.1 vs. capitalized in Ch06. Standardize.
- CE-25: backtick formatting around k inconsistent — sometimes `k`, sometimes plain text. Use $k$ (LaTeX).
- CE-26: "a 10+ point gap" — the exact gap is 10.0 percentage points. Say that.
- CE-27: "In the computer science sense" before BM25 description adds nothing. Remove.
- CE-28: Opening epigraph is a bibliographic reference, not a quotable line. Find a quotable line.

### Ch08
- CE-29: "otherwise the signal would grow without bound" — it's the variance, not the signal.
- CE-30: Song et al. "(2020)" in text but references list says "ICLR 2021." Match conventions.
- CE-31: "The VAE's encoder is not lossy in the semantic sense" — all compression is lossy. Qualify.
- CE-32: "a second copy of the U-Net encoder" — ControlNet copies encoder blocks, not the full encoder.
- CE-33: DreamBooth mentioned in Connections but never explained. Either describe briefly or remove.

### Ch09
- CE-34: "An A100 80GB GPU can barely hold it" — it literally can't (84 > 80). Say "exceeds the memory."
- CE-35: "You 'train with adapters, deploy without them'" — quotation marks imply attribution. Either attribute or remove quotes.
- CE-36: Model name "meta-llama/Llama-2-7b-hf" is gated and outdated by 2026. Add comment or update.
- CE-37: Missing `import torch` and `from transformers import Trainer` in code.
- CE-38: "BERT-base has 12 transformer layers, each getting 2 adapters" — this describes Houlsby configuration only. Qualify.

### Ch10
- CE-39: "self-consistency... improves on chain-of-thought" — say "chain-of-thought prompting."
- CE-40: "Before CLIP, vision models were trained on fixed label sets" — oversimplifies. Say "the dominant paradigm was..."
- CE-41: "a Vicuna LLM (a fine-tuned LLaMA)" — standardize LLaMA vs. Llama across the book.
- CE-42: Video-LLaVA and VideoChat mentioned without citations.
- CE-43: DeepSeek-R1 discussed in Section 10.3 but missing from references. Add.
- CE-44: Winoground and ARO benchmarks mentioned without citations.

---

## Part 7: Frontmatter-Specific Issues

- FM-1: Core Papers table numbering doesn't match chapter order. Either renumber by chapter or drop the numbering column.
- FM-2: "Over 80 peer-reviewed papers" — some cited works are arXiv preprints. The closing note says "All references are to peer-reviewed publications or established arXiv preprints." Be consistent — either drop the "peer-reviewed" count or say "80+ referenced papers."
- FM-3: Rombach et al. chapter title says "Stable Diffusion" but the paper is about Latent Diffusion Models. Stable Diffusion is a product built on the paper.
- FM-4: Subtitle too long. Consider moving "For the Engineer Who Builds Systems and Wants to Understand the Science Behind Them" into the preface.
- FM-5: Acronyms (ELBO, RLHF, DPO, RETRO, Self-RAG) unexpanded in ToC descriptions.
- FM-6: "Generated March 2026" — use "Last updated March 2026" or a copyright date.
- FM-7: Xu et al. (2023) PEFT survey — multiple exist. Provide full title for disambiguation.
