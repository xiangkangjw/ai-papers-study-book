# How Do I Know It's Working? — Evaluation and Debugging Guide

This guide provides per-chapter evaluation criteria, failure mode recognition, and debugging tips. Use it alongside the main chapters when implementing and training the models covered in this book.

---

## Ch02 — Variational Autoencoders (VAE)

**Key Metrics**
- **Reconstruction loss**: MSE or binary cross-entropy between input and reconstruction. Should decrease steadily during training.
- **KL divergence term**: Measures how far the posterior q(z|x) is from the prior p(z). A healthy value is neither near zero nor explosively large.
- **ELBO**: The sum of reconstruction loss and KL term. Track both components separately, not just the total.
- **FID (Fréchet Inception Distance)**: For image VAEs, FID on held-out samples gives a generation quality signal independent of reconstruction.

**Common Failure Modes**
- **Posterior collapse**: The KL term converges to zero and the decoder ignores z entirely, producing blurry averages. Symptom: KL ≈ 0 after a few epochs, reconstruction looks like dataset mean.
- **Blurry reconstructions**: Over-weighting the KL term relative to reconstruction pushes posteriors toward the prior too aggressively. Try reducing the β coefficient in β-VAE.
- **Holes in latent space**: Samples from p(z) look nothing like real data even though reconstructions look fine. The encoder has learned a non-smooth posterior that doesn't cover the prior well.

**Debugging Tips**
- Monitor KL per latent dimension. If most dimensions collapse to zero, you have posterior collapse — try KL annealing (warm up the KL weight from 0 over the first N epochs).
- Visualize reconstructions from the first epoch onward. If they are blurry throughout, check the reconstruction loss scale — it should dominate early training.
- For posterior collapse in sequence models: try free bits (set a minimum KL per dimension below which no gradient flows).
- Plot the 2D latent space (or a 2D PCA projection) and color by class label. A healthy VAE shows smooth, overlapping clusters.

---

## Ch03 — Generative Adversarial Networks (GAN)

**Key Metrics**
- **FID (Fréchet Inception Distance)**: Primary metric for image generation quality. Lower is better. Typical good values: <10 for class-conditional ImageNet models, <5 for state-of-the-art.
- **Inception Score (IS)**: Measures both quality and diversity. Higher is better, but IS can be gamed and does not penalize mode dropping as severely as FID.
- **Precision and Recall**: Precision measures sample quality (how many generated samples fall in the real data manifold); Recall measures diversity (how much of the real manifold is covered).
- **Discriminator and generator loss**: Track separately. Neither should go to zero.

**Common Failure Modes**
- **Mode collapse**: Generator produces a small subset of plausible outputs — often one or a few modes. Symptom: FID is mediocre but IS looks acceptable; generated samples cluster visually.
- **Discriminator dominates**: Generator loss spikes or diverges. The discriminator has learned too fast and is providing vanishing gradients to the generator.
- **Training instability**: Oscillating losses, NaN gradients, or outputs that degrade suddenly after appearing to improve.
- **Checkerboard artifacts**: Particularly in convolutional generators using transposed convolutions. Switch to resize-then-convolve upsampling.

**Debugging Tips**
- If discriminator loss drops to near zero quickly, add label smoothing (real labels = 0.9 instead of 1.0) or reduce discriminator learning rate.
- Gradient penalty (WGAN-GP) stabilizes training significantly at the cost of compute. If vanilla GAN is unstable, this is usually the first fix to try.
- Visualize a fixed batch of latent vectors across checkpoints. If the outputs stop changing, you have mode collapse.
- Track the discriminator's accuracy on a held-out real/fake set. Healthy: roughly 50–70%. If it reaches 99%, something is wrong.
- For mode collapse, try minibatch discrimination or feature matching as auxiliary objectives.

---

## Ch04 — The Transformer

**Key Metrics**
- **Perplexity**: For language modeling heads, perplexity on a held-out set is the primary metric. Lower is better.
- **Training and validation loss curves**: Both should decrease; a growing gap indicates overfitting.
- **Gradient norms**: Per-layer and global gradient norms. Should be stable and not exploding.
- **Attention entropy**: Average entropy of attention distributions per head. Very low entropy (sharp attention) or very high entropy (diffuse, near-uniform) can both indicate problems.

**Common Failure Modes**
- **Exploding or vanishing gradients**: Especially in deep Transformers without careful initialization. Symptom: loss diverges or stalls in the first few hundred steps.
- **Attention collapse**: All heads attend to the same positions (e.g., always the [CLS] token or the first token). The model is not using multi-head attention effectively.
- **Slow warmup**: Learning rate too high before the model has stable representations. Always use a warmup schedule (linear or cosine with warmup).
- **Positional encoding mismatch**: Applying relative positional encodings inconsistently across layers, or forgetting to add positional encodings entirely.

**Debugging Tips**
- Visualize attention patterns with a heatmap tool (e.g., BertViz). Look for at least some heads showing local attention, some showing global, some showing syntactic patterns.
- If loss diverges in the first 1000 steps, reduce peak learning rate or increase warmup steps.
- Check that LayerNorm is applied consistently (pre-norm vs. post-norm — be consistent with your reference implementation).
- For training stability, gradient clipping at norm 1.0 is standard. If you're not clipping, add it.
- Confirm that the causal mask is correctly applied for decoder-style (autoregressive) models — an unmasked decoder will see future tokens and achieve perfect training loss but fail at inference.

---

## Ch05 — Pre-trained Language Models (GPT and BERT)

**Key Metrics**
- **Pre-training perplexity**: Lower is better; compare to published checkpoints of similar scale as a sanity check.
- **Downstream task metrics**: GLUE/SuperGLUE for BERT-style models; few-shot accuracy on standard benchmarks (HellaSwag, MMLU, ARC) for GPT-style models.
- **In-context learning (ICL) performance**: For GPT-style models, track how performance scales with the number of in-context examples.
- **Tokenization edge cases**: Check model behavior on numbers, code, non-English text, and rare words that may be over-fragmented by the tokenizer.

**Common Failure Modes**
- **Repetition and degeneration**: GPT-style models looping or producing repetitive output. Common with greedy decoding; use nucleus sampling (top-p) or temperature scaling.
- **BERT fine-tuning instability**: Fine-tuning a BERT checkpoint on small datasets often leads to high variance across runs. Use a small learning rate (2e-5), weight decay, and run multiple seeds.
- **Catastrophic forgetting during fine-tuning**: The model loses general language ability when fine-tuned aggressively. Use a lower learning rate for earlier layers.
- **Tokenizer mismatch**: Using a fine-tuned model with the wrong tokenizer. Always save and load the tokenizer alongside the checkpoint.

**Debugging Tips**
- For GPT fine-tuning, probe the model with held-out prompts from the original pre-training distribution to check whether general capabilities are degrading.
- For BERT, if fine-tuning accuracy is erratic across runs, try a learning rate warmup over 6–10% of training steps and gradient clipping.
- Inspect tokenized inputs manually for your target domain. If technical vocabulary is being split into many subword pieces, consider domain-adaptive pre-training or a domain-specific tokenizer.
- Track train/validation loss separately. Overfitting is common on small fine-tuning sets — early stopping or a dropout increase may be needed.
- For in-context learning: ensure consistent prompt format. Even minor formatting differences between few-shot examples and the query can cause significant performance drops.

---

## Ch06 — Vision Transformer (ViT)

**Key Metrics**
- **Top-1 and Top-5 accuracy**: Standard ImageNet metrics. Compare to published ViT baselines at the same model size and training budget.
- **Data efficiency curves**: Plot validation accuracy vs. number of training examples. ViT underperforms CNNs in low-data regimes; this curve reveals whether you're in that regime.
- **Attention distance**: Average distance (in patch units) attended to per head per layer. Early layers should have short-range attention; later layers should have longer-range attention.
- **Patch embedding quality**: Visualize learned patch embedding filters. They should resemble oriented edge detectors in early layers (analogous to CNN first-layer filters).

**Common Failure Modes**
- **Poor performance at small dataset scale**: ViT needs either a large dataset or aggressive augmentation (RandAugment, MixUp, CutMix). Without this, ResNet-family models will outperform ViT at the same parameter count.
- **High sensitivity to learning rate**: ViT training is more sensitive to learning rate than CNNs. Small changes can cause instability.
- **Uniform attention maps**: If attention maps look diffuse and non-selective throughout all layers, the model may not be learning useful spatial features.
- **Patch resolution mismatch at inference**: Fine-tuning at a different resolution than pre-training requires interpolating positional embeddings — skipping this step degrades performance.

**Debugging Tips**
- Use DeiT-style data augmentation and distillation if training on ImageNet-scale data without JFT-scale pre-training.
- Visualize attention rollout (aggregating attention across layers) to confirm the model is attending to semantically relevant regions.
- If training is unstable, try a lower peak learning rate (3e-4 or less) with cosine decay and linear warmup over 5–10 epochs.
- For fine-tuning at a new resolution, always interpolate positional embeddings using 2D bilinear interpolation of the original grid.
- MAE pre-training (Ch06 section on masked autoencoders) significantly improves data efficiency — consider it as a pre-training step before supervised fine-tuning.

---

## Ch07 — Retrieval-Augmented Generation (RAG)

**Key Metrics**
- **Recall@k**: Of all relevant documents for a query set, what fraction appear in the top-k retrieved results? This is the primary retrieval quality metric.
- **RAGAS metrics**: A suite designed for end-to-end RAG evaluation: faithfulness (does the answer follow from the context?), answer relevancy (does the answer address the question?), context precision, and context recall.
- **Retrieval latency**: P50 and P99 latency for the retrieval step at your target query volume. Vector search should be <100ms for most production use cases.
- **Chunk quality**: Human or LLM-rated assessment of whether chunks contain complete, coherent units of information rather than fragmented sentences.

**Common Failure Modes**
- **Retrieval misses**: The correct document is not in the top-k, so the LLM has no chance of answering correctly. Symptom: high faithfulness but low recall@k; answers are confident but wrong.
- **Context window overflow**: Retrieved chunks plus query exceed the LLM's context window. Symptom: hard truncation or API errors at inference time.
- **Chunk fragmentation**: Splitting documents on fixed character boundaries breaks sentences and severs meaning. The retrieved chunk lacks the context needed to answer the question.
- **Semantic drift in embeddings**: The embedding model was trained on a different domain than your documents. Retrieval returns topically adjacent but not precisely relevant chunks.

**Debugging Tips**
- Start with a retrieval-only evaluation: for a test set of question-answer pairs, check whether the correct document is in the top-5 retrieved results. Fix retrieval before fixing generation.
- If recall@k is low, try: (1) smaller chunk sizes with overlap, (2) a stronger embedding model (e.g., BGE-M3 or Cohere embed), (3) hybrid search (dense + BM25 sparse), (4) a reranker (e.g., CrossEncoder) on top of initial retrieval.
- Log the retrieved chunks alongside the final answer during development. This lets you quickly see whether failures are retrieval failures or generation failures.
- For chunk quality: prefer semantic chunking (split on paragraph or section boundaries) over fixed-size chunking.
- Test your system on adversarial queries: questions whose answer spans multiple documents, questions with no answer in the corpus, and questions whose keywords appear in irrelevant documents.

---

## Ch08 — Diffusion Models and Stable Diffusion

**Key Metrics**
- **FID (Fréchet Inception Distance)**: Primary metric for unconditional and class-conditional generation. Lower is better.
- **CLIP score**: For text-to-image models, measures alignment between generated image and the conditioning text prompt. Higher is better (range roughly 20–35 for current models).
- **LPIPS (Learned Perceptual Image Patch Similarity)**: For image-to-image tasks (inpainting, super-resolution), measures perceptual similarity. Lower is better.
- **Human evaluation**: For text-to-image, automated metrics often miss fine-grained prompt adherence. Include a human side-by-side evaluation for important comparisons.

**Common Failure Modes**
- **Artifacts from aggressive CFG**: Classifier-free guidance (CFG) scale too high produces oversaturated, overcontrasted images with burned highlights and clipped shadows. Typical sweet spot: 7–12 for most prompts.
- **Anatomy failures**: Incorrect number of limbs, deformed hands, inconsistent geometry. A persistent limitation of current diffusion models; worse on complex scenes.
- **Color bleeding and semantic confusion**: Separate concepts in a prompt bleeding into each other spatially. Use explicit spatial prompting or ControlNet conditioning.
- **Slow convergence during fine-tuning**: Full fine-tuning of a diffusion U-Net is expensive. Prefer DreamBooth or LoRA fine-tuning for concept injection.

**Debugging Tips**
- If outputs look overprocessed or surreal, reduce CFG scale first. This is the most common cause of visible artifacts.
- For DDPM training, plot the noise prediction loss (epsilon prediction MSE) by noise level. It should be roughly uniform across timesteps; if high-noise timesteps have much higher loss, consider a min-SNR weighting schedule.
- Visualize the denoising trajectory (intermediate samples at T=900, 700, 500, 300, 100) to understand where the model is committing to structure vs. details.
- For latent diffusion: check that your VAE encoder and decoder are functioning correctly before training the diffusion model. Reconstruction quality of the VAE sets a ceiling on final image quality.
- CLIP score is noisy for individual samples — always evaluate on hundreds of samples with diverse prompts.

---

## Ch09 — Parameter-Efficient Fine-Tuning (LoRA / PEFT)

**Key Metrics**
- **Task performance vs. full fine-tune delta**: Compare your LoRA-fine-tuned model to a fully fine-tuned baseline on the target task. A well-tuned LoRA should reach 90–98% of full fine-tune performance.
- **Rank sensitivity**: Evaluate task performance at rank r = 1, 2, 4, 8, 16, 32. Plot the curve — most tasks saturate well before r = 64.
- **Parameter count and memory footprint**: Track trainable parameters as a fraction of total model parameters. LoRA typically achieves <1% trainable parameters.
- **Inference latency**: After merging LoRA weights back into the base model, latency should be identical to the base model. If using adapters that are not merged, measure the overhead.

**Common Failure Modes**
- **Catastrophic forgetting**: Even with PEFT, aggressive fine-tuning on narrow tasks can degrade general capabilities. Symptom: target task improves but performance on unrelated benchmarks drops significantly.
- **Rank too low**: Under-parameterized LoRA (e.g., r=1) on complex tasks fails to capture the necessary adaptation. Task performance plateaus far below the full fine-tune baseline.
- **Applying LoRA to wrong layers**: By default, LoRA targets attention weight matrices (q_proj, v_proj). For some tasks (e.g., code generation), also targeting the feed-forward layers improves results.
- **Alpha/rank ratio mismatch**: The LoRA scaling factor α/r controls effective learning rate for the adapter. If α = r (common default), the effective contribution is 1.0; if α >> r, adapter updates dominate the base weights.

**Debugging Tips**
- Always evaluate on a general-purpose benchmark (e.g., MMLU or a held-out general QA set) alongside the target task to monitor for forgetting.
- If performance is below the full fine-tune baseline, increase rank before increasing learning rate.
- For QLoRA: ensure the base model is quantized with NF4 (not INT8) and double quantization is enabled for best memory savings. Verify that the compute dtype is bfloat16 to avoid precision loss during adapter updates.
- Compare rank sensitivity curves across task types. Tasks requiring narrow factual memorization often need low rank; tasks requiring broad style or behavior shifts need higher rank.
- After training, merge the LoRA weights and verify the merged checkpoint matches the adapter-loaded checkpoint on a reference prompt. Mismatches indicate a merge implementation bug.

---

## Ch10 — Reinforcement Learning and Vision-Language Models (RL / VLM)

**Key Metrics**
- **Reward curve**: Average reward per episode (RL) or per sample (RLHF). Should increase over training. A flat or declining reward curve early on indicates insufficient exploration or a poorly specified reward model.
- **KL divergence from reference policy**: In RLHF/PPO, the KL between the trained policy and the reference (SFT) policy is constrained by a coefficient β. Track this: it should stabilize, not grow unboundedly.
- **Win rate vs. SFT baseline**: For RLHF-trained models, the primary evaluation is human preference win rate against the supervised fine-tuning baseline.
- **VQA accuracy**: For Vision-Language Models evaluated on visual question answering benchmarks (VQA v2, GQA, MMBench). Broken down by question type (yes/no, number, open-ended).
- **Hallucination rate**: For VLMs, the fraction of responses that assert visually unverifiable or incorrect facts about the image.

**Common Failure Modes**
- **Reward hacking**: The policy finds behaviors that score highly on the reward model but are clearly undesirable to humans. Symptom: reward increases but human evaluators rate outputs as worse.
- **KL divergence explosion**: The policy drifts too far from the reference, losing coherence. Symptom: outputs become repetitive, grammatically broken, or semantically empty despite high reward.
- **Mode collapse in RL**: Policy converges to a narrow set of response patterns. Increases reward on the training distribution but reduces diversity and generalization.
- **VLM hallucination**: The language model component generates plausible-sounding text that ignores or misinterprets the visual input. Common in models where the vision encoder and language model are poorly aligned.
- **Sparse reward signal**: In RL environments with infrequent rewards, the agent fails to learn because most trajectories receive no signal. Use reward shaping or curriculum learning to address this.

**Debugging Tips**
- Plot reward, KL divergence, and policy entropy simultaneously. A healthy RLHF run shows reward increasing, KL stabilizing within the target range, and entropy not collapsing.
- If KL diverges, increase the KL penalty coefficient β or reduce the policy learning rate.
- For reward hacking: periodically sample outputs manually and look for pattern exploitation. Consider training an adversarial reward model or using a held-out reward model for evaluation.
- For VLMs: evaluate attention to visual tokens separately from language generation quality. If the model is not attending to the image, the problem is in the vision-language connector, not the language model.
- Use DPO (Direct Preference Optimization) as an alternative to PPO-based RLHF when you have preference data but want to avoid the instability of online RL training. DPO is significantly easier to tune and debug.
- For VQA debugging: break down errors by question type and image category. Models often fail systematically on counting, spatial reasoning, or text-in-image questions — knowing which category helps target architectural fixes.
