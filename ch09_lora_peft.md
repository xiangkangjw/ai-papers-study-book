# Chapter 9: Parameter-Efficient Fine-Tuning — Adapting Giants Without Breaking the Bank

> **Key Papers**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"; Xu et al. (2023) "Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment"

---

## 9.1 The Fine-Tuning Problem: A Systems Engineering Perspective

Suppose you have a 7B-parameter language model — say, LLaMA-2-7B. You want to specialize it for five different tasks: SQL generation, medical QA, legal document summarization, code completion, and customer support. The naive approach is full fine-tuning: for each task, load the model, update all 7 billion parameters, and save the result.

Let's work out what that actually costs.

**Memory requirements for full fine-tuning** (mixed-precision, AdamW optimizer):

| Component | Per-parameter cost | Total (7B model) |
|---|---|---|
| Master weights (fp32) | 4 bytes | 28 GB |
| Model weights (fp16, for forward pass) | 2 bytes | 14 GB |
| Gradients (fp16) | 2 bytes | 14 GB |
| Optimizer states — first moment (fp32) | 4 bytes | 28 GB |
| Optimizer states — second moment (fp32) | 4 bytes | 28 GB |
| **Total** | **16 bytes** | **~112 GB** |

That's 112 GB just for training one variant — well beyond the 80 GB capacity of an A100 GPU (requiring model parallelism or offloading), and you need it entirely for that one task. If you want to serve all five task-specific models in production, you need to store $5 \times 14 \text{ GB} = 70 \text{ GB}$ of model weights alone, plus infrastructure to load/unload them per request.

This is the **deployment scaling problem**. It has three distinct sub-problems:

1. **Training cost**: You cannot fit a large model's optimizer states in a single GPU's memory.
2. **Storage cost**: Each fine-tuned variant requires a full copy of the model.
3. **Serving cost**: Switching between task variants requires loading different model weights, causing latency spikes or requiring dedicated replicas per task.

Parameter-Efficient Fine-Tuning (PEFT) attacks all three. The core idea: freeze the pre-trained weights, add a small number of trainable parameters, and train only those. The pre-trained model becomes a shared backbone — one copy, many adapters.

---

## 9.2 The PEFT Landscape

Xu et al. (2023) survey over 40 PEFT methods and organize them into three families:

```
PEFT Methods
├── Additive Methods
│   ├── Adapters (Houlsby et al., 2019)
│   ├── Prefix Tuning (Li & Liang, 2021)
│   └── Prompt Tuning (Lester et al., 2021)
├── Selective Methods
│   ├── BitFit (Ben Zaken et al., 2022)
│   └── Diff Pruning (Guo et al., 2021)
└── Reparameterization Methods
    ├── LoRA (Hu et al., 2021)
    ├── AdaLoRA (Zhang et al., 2023)
    └── KronA (Edalati et al., 2022)
```

**Additive methods** insert new parameters into the model architecture or prepend learnable tokens to the input. The original weights remain frozen; only the new components are trained.

**Selective methods** identify a subset of the existing parameters to train, leaving everything else frozen. BitFit, for example, trains only the bias terms — roughly 0.1% of parameters for BERT-scale models.

**Reparameterization methods** constrain the weight update to a lower-dimensional form. Rather than adding new components, they express $\Delta \mathbf{W}$ as a product of smaller matrices. LoRA is the canonical example.

We'll cover additive methods briefly, then spend most of this chapter on LoRA — which has become the dominant approach in practice.

---

## 9.3 Additive Methods

### 9.3.1 Adapters

Houlsby et al. (2019) introduced the adapter pattern: insert small bottleneck modules between transformer layers.

An adapter takes the hidden state $\mathbf{h} \in \mathbb{R}^d$, projects it down to a bottleneck dimension $m \ll d$, applies a nonlinearity, and projects back up:

$$\text{Adapter}(\mathbf{h}) = \mathbf{h} + \mathbf{W}_{\text{up}} \cdot \sigma(\mathbf{W}_{\text{down}} \cdot \mathbf{h})$$

$$\text{where } \mathbf{W}_{\text{down}} \in \mathbb{R}^{m \times d},\ \mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times m}$$

The residual connection ensures that at initialization (if $\mathbf{W}_{\text{up}}$ is zero), the adapter is an identity function. Only $\mathbf{W}_{\text{down}}$ and $\mathbf{W}_{\text{up}}$ are trained.

**Parameter count**: $2dm$ per adapter. With $d = 768$ (BERT-base) and $m = 64$, that's 98,304 parameters per adapter. BERT-base has 12 transformer layers, each getting 2 adapters (one after self-attention, one after FFN), so total added parameters: $24 \times 98{,}304 \approx 2.4\text{M}$ out of BERT's 110M total — about 2%.

**The problem**: Adapters add layers to the forward pass. At inference, you cannot remove them — unlike LoRA. This creates nonzero latency overhead, which matters at scale.

### 9.3.2 Prefix Tuning

Li & Liang (2021) take a different approach: instead of modifying the architecture, prepend learnable "virtual tokens" directly into the key and value tensors of each attention layer.

In standard attention, for a sequence of length $n$:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

With prefix tuning, you prepend $L$ learnable prefix vectors to $\mathbf{K}$ and $\mathbf{V}$:

$$\mathbf{K}' = [\mathbf{P}_K;\ \mathbf{K}], \quad \mathbf{V}' = [\mathbf{P}_V;\ \mathbf{V}] \quad \text{where } \mathbf{P}_K, \mathbf{P}_V \in \mathbb{R}^{L \times d}$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}', \mathbf{V}') = \text{softmax}\!\left(\frac{\mathbf{Q}[\mathbf{P}_K;\ \mathbf{K}]^\top}{\sqrt{d_k}}\right) [\mathbf{P}_V;\ \mathbf{V}]$$

The prefix tokens attend to (and are attended to by) all input tokens, effectively steering the model's behavior through the key-value space. Think of it as **soft prompting in the activation space** — rather than modifying discrete input tokens, you're inserting continuous, differentiable "instructions" directly where attention operates.

**Practical subtlety**: The prefix parameters are highly sensitive to initialization and training is unstable. Li & Liang reparameterize: the actual prefix is generated by $\text{MLP}(\tilde{\mathbf{P}})$ where $\tilde{\mathbf{P}}$ is smaller; the MLP is discarded after training, leaving only the final prefix values.

### 9.3.3 Prompt Tuning

Lester et al. (2021) simplify prefix tuning to its most minimal form: just prepend learnable embeddings to the input token embeddings, and train nothing else.

```
Input:  [p_1, ..., p_k, x_1, ..., x_n]   (k soft prompt tokens + n real tokens)
Frozen: everything in the transformer
Trained: {p_1, ..., p_k}
```

At small model sizes (< 1B parameters), prompt tuning lags behind full fine-tuning. But as models scale to 10B+ parameters, the gap nearly closes. The explanation: large models are already highly capable; the soft prompt only needs to steer, not teach.

**The key insight from Lester et al.**: Prompt tuning gets more parameter-efficient as model size grows. For very large models, a handful of soft prompt tokens suffice.

---

## 9.4 LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) is where the field largely converged. It achieves the best combination of: competitive performance, zero inference latency overhead, clean theoretical motivation, and composability. Let's build it from scratch.

### 9.4.1 The Core Insight: Fine-Tuning is Low-Rank

Aghajanyan et al. (2021) showed empirically that when you fine-tune a large language model, the weight updates have **low intrinsic rank**. Concretely: during fine-tuning, the change $\Delta \mathbf{W} = \mathbf{W}_{\text{fine-tuned}} - \mathbf{W}_{\text{pre-trained}}$ can be well-approximated by a rank-$r$ matrix, where $r$ is much smaller than the full rank of $\mathbf{W}$.

This is not obvious — gradient descent on a large matrix doesn't have to produce low-rank updates. But it turns out the optimization landscape of fine-tuning tends to concentrate updates in a low-dimensional subspace. The intuition: the pre-trained model already captures most of the relevant structure; fine-tuning only needs to shift the representation slightly.

If $\Delta \mathbf{W}$ is approximately low-rank, we can write:

$$\Delta \mathbf{W} \approx \mathbf{B} \mathbf{A} \quad \text{where } \mathbf{B} \in \mathbb{R}^{d \times r},\ \mathbf{A} \in \mathbb{R}^{r \times k},\ \text{rank}(\mathbf{B}\mathbf{A}) \leq r$$

Rather than storing and updating the full $d \times k$ matrix $\Delta \mathbf{W}$, we only need to store and update $\mathbf{B}$ and $\mathbf{A}$.

### 9.4.2 The Math

For a weight matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$ (say, the query projection in attention), the standard forward pass is:

$$\mathbf{h} = \mathbf{W} \mathbf{x}$$

With LoRA, the forward pass becomes:

$$\mathbf{h} = \mathbf{W} \mathbf{x} + (\mathbf{B} \mathbf{A}) \mathbf{x} = \mathbf{W} \mathbf{x} + \mathbf{B} (\mathbf{A} \mathbf{x})$$

where:
- $\mathbf{W} \in \mathbb{R}^{d \times k}$ is frozen (pre-trained weights, never updated)
- $\mathbf{A} \in \mathbb{R}^{r \times k}$ is trainable
- $\mathbf{B} \in \mathbb{R}^{d \times r}$ is trainable
- $r \ll \min(d, k)$ is the rank hyperparameter

**Initialization**:
- $\mathbf{A}$ is initialized from $\mathcal{N}(0, \sigma^2)$ (random Gaussian)
- $\mathbf{B}$ is initialized to zero

At the start of training, $\mathbf{B}\mathbf{A} = \mathbf{0}$, so the modified model is identical to the pre-trained model. This is crucial: LoRA training starts from exactly the pre-trained behavior, not from some random perturbation.

**Scaling factor**: The LoRA output is scaled by $\alpha/r$ where $\alpha$ is a hyperparameter:

$$\mathbf{h} = \mathbf{W} \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$$

This scaling decouples the effective learning rate from the rank. When you change $r$, you don't need to re-tune $\alpha$. In practice, $\alpha$ is often set to $r$ (giving a scaling factor of 1) or to a fixed value like 16 regardless of $r$.

### 9.4.3 Parameter Count and Memory Analysis

Consider a weight matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$ (square, for simplicity). Full fine-tuning requires storing $\Delta \mathbf{W} \in \mathbb{R}^{d \times d}$. With LoRA at rank $r$:

| | Full fine-tuning | LoRA (rank $r$) |
|---|---|---|
| Trainable parameters | $d^2$ | $2dr$ |
| Optimizer first moment | $d^2$ fp32 values | $2dr$ fp32 values |
| Optimizer second moment | $d^2$ fp32 values | $2dr$ fp32 values |
| Gradient storage | $d^2$ fp16 values | $2dr$ fp16 values |

For $d = 4096$ (LLaMA-7B attention dimension) and $r = 8$:
- Full: $4096^2 = 16{,}777{,}216$ parameters per matrix
- LoRA: $2 \times 4096 \times 8 = 65{,}536$ parameters per matrix
- **Reduction factor**: $256\times$

LLaMA-7B has 32 transformer layers, each with 4 attention weight matrices (Q, K, V, O). Applying LoRA with $r = 8$ to all of them:

| | Full fine-tuning (attn weights only) | LoRA $r=8$ (attn weights only) |
|---|---|---|
| Trainable params | $32 \times 4 \times 16.8\text{M} = 2.15\text{B}$ | $32 \times 4 \times 65{,}536 = 8.4\text{M}$ |
| Optimizer states (fp32) | ~17 GB | ~67 MB |
| Adapter storage | 14 GB (full weights) | 34 MB |

This is the key result: **LoRA reduces trainable parameters by ~$256\times$ and optimizer state memory by the same factor.** The 14 GB base model is shared across all tasks; each task-specific adapter is only ~34 MB.

### 9.4.4 Which Layers to Adapt

Hu et al. found that applying LoRA to $\mathbf{W}_q$ and $\mathbf{W}_v$ (query and value projections) in the attention layers achieves the best performance-to-parameter ratio. Applying it to all four attention matrices ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$) improves performance further at the cost of more parameters. Applying LoRA to the feed-forward layers (the MLP blocks) is also possible and sometimes beneficial.

The survey by Xu et al. confirms this: attention weights are generally the highest-value targets, but for tasks requiring strong reasoning or factual recall, including the FFN layers helps. For instruction-following and style transfer, attention-only LoRA usually suffices.

### 9.4.5 Merge at Inference: Zero Latency

This is LoRA's killer feature relative to adapters. At inference time, you can merge the LoRA parameters back into the base weights:

$$\mathbf{W}_{\text{merged}} = \mathbf{W} + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

After merging, $\mathbf{B}$ and $\mathbf{A}$ are discarded. The inference graph is identical to the original model — same architecture, same compute graph, same latency. You "train with adapters, deploy without them."

The merge is a single matrix addition per adapted layer. It's fast and exact. No approximation.

**Implication**: At serving time, you can either:
1. **Pre-merge**: Merge adapters into a single model file. Zero overhead, but you need a separate model file per task.
2. **Dynamic application**: Keep the base model frozen and apply adapters on the fly. Adds a small matrix multiply per adapted layer, but you can hot-swap adapters.

Option 2 is what S-LoRA (Sheng et al., 2023) optimizes — more on this in Section 9.6.

---

## 9.5 LoRA Variants

### 9.5.1 QLoRA: Quantize the Base, LoRA the Updates

Dettmers et al. (2023) observe that LoRA cuts the optimizer state memory, but the frozen base model still needs to fit in GPU memory. For a 65B-parameter model, that's ~130 GB in fp16 — requiring multiple high-end GPUs just to load.

QLoRA's solution: quantize the frozen base model to 4 bits, and apply LoRA on top. The key innovations:

**NF4 (NormalFloat4)**: A 4-bit data type designed for normally-distributed weights. Unlike naive 4-bit quantization (which uniformly spaces quantization bins), NF4 places bins at the quantiles of a standard normal distribution. This minimizes quantization error for neural network weights, which are empirically near-normal.

**Double quantization**: The quantization constants themselves (one per small block of weights) are also quantized, saving an additional ~0.5 bits per parameter.

**Paged optimizers**: When GPU memory is full during a backward pass, optimizer states are evicted to CPU RAM and paged back in when needed — borrowing the OS paging concept.

**Result**: A 65B model requires ~35 GB to train with QLoRA, fitting on a single A100 80GB GPU. Without QLoRA, you'd need at least $4\times$ A100s.

**The catch**: At inference, the quantized base model must be dequantized before the LoRA adapter output is added. Recent versions of the PEFT library support `merge_and_unload()` for QLoRA, which dequantizes, merges the adapter, and returns a full-precision model. If you want quantized inference after merging, you can then requantize the merged model.

### 9.5.2 LoRA+: Asymmetric Learning Rates

Hayou et al. (2024) identify a subtle inefficiency in standard LoRA: the two matrices $\mathbf{A}$ and $\mathbf{B}$ play asymmetric roles (one is initialized to random, one to zero), but they're typically trained with the same learning rate.

LoRA+ uses a higher learning rate for $\mathbf{B}$ than for $\mathbf{A}$ — roughly $\eta_B = \lambda \eta_A$ where $\lambda$ is typically in the range 4–16 (varying by model and task). The theoretical justification comes from the analysis of feature learning in neural networks: $\mathbf{B}$ needs to move faster to effectively leverage the features learned by $\mathbf{A}$.

Empirically, LoRA+ achieves the same final performance as standard LoRA in fewer training steps, or better final performance with the same steps.

### 9.5.3 DoRA: Decompose Magnitude and Direction

Liu et al. (2024) note that a weight matrix $\mathbf{W}$ can be decomposed into magnitude and direction:

$$\mathbf{W} = \mathbf{m} \cdot \frac{\mathbf{W}}{\|\mathbf{W}\|_c} \quad \text{where } \mathbf{m} = \|\mathbf{W}\|_c \text{ is a per-column norm vector}$$

Standard LoRA adapts both magnitude and direction simultaneously, which may cause the learning to interfere with itself. DoRA (Weight-Decomposed Low-Rank Adaptation) applies LoRA only to the directional component:

$$\mathbf{W}' = (\mathbf{m} + \Delta \mathbf{m}) \cdot \frac{\mathbf{W} + \mathbf{B}\mathbf{A}}{\|\mathbf{W} + \mathbf{B}\mathbf{A}\|_c}$$

Where $\Delta \mathbf{m}$ is a learned scalar per column and $\mathbf{B}\mathbf{A}$ is the standard LoRA update on the directional component. The magnitude can be updated freely (it's cheap — just one scalar per column), and the directional update is constrained to a low-rank subspace.

DoRA consistently outperforms LoRA at the same parameter count, particularly on tasks requiring precise factual recall.

### 9.5.4 AdaLoRA: Adaptive Rank Allocation

Standard LoRA uses the same rank $r$ for every adapted layer. This is a blunt instrument — some layers may benefit from higher rank while others need minimal adaptation.

AdaLoRA (Zhang et al., 2023) formulates rank allocation as a budget-constrained optimization. It parameterizes the update as an SVD-like decomposition:

$$\Delta \mathbf{W} = \mathbf{P} \mathbf{\Lambda} \mathbf{Q} \quad \text{where } \mathbf{P} \in \mathbb{R}^{d \times r},\ \mathbf{Q} \in \mathbb{R}^{r \times k},\ \mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_r)$$

During training, it estimates the importance of each singular value $\lambda_i$ using an importance score derived from gradient magnitudes. Unimportant singular vectors are pruned (set to zero), effectively reducing the rank of less important layers while maintaining higher rank where needed.

AdaLoRA requires a target parameter budget and automatically distributes rank across layers. It typically outperforms fixed-rank LoRA at the same total parameter count.

---

## 9.6 Practical Guide

### 9.6.1 When to Use Which Method

| Scenario | Recommended approach |
|---|---|
| Access to full GPU cluster, needs maximum performance | Full fine-tuning |
| Single A100 (80GB), model $\leq$ 13B | LoRA (fp16 base) |
| Single A100 (80GB), model 13B–70B | QLoRA |
| Multiple tasks, need hot-swappable adapters | LoRA with dynamic application |
| Extreme parameter efficiency, large model | Prompt tuning (>10B base model) |
| Tasks requiring precise factual recall | DoRA or AdaLoRA |
| Unstable training, want careful control | AdaLoRA or LoRA+ |

### 9.6.2 LoRA Hyperparameter Selection

**Rank $r$**: The most important hyperparameter. Higher rank = more trainable parameters = more adaptation capacity.
- $r = 4$: Very lightweight; works for style/tone changes, instruction following
- $r = 8$: Good default for most tasks
- $r = 16$–$32$: More complex tasks, domain adaptation
- $r = 64+$: Approaching full fine-tuning territory; diminishing returns often appear by $r=64$ for many tasks, though complex reasoning tasks (e.g., mathematical problem solving) can benefit from $r=128$ or higher

The relationship between $r$ and task complexity is empirical, not theoretical. When in doubt, start at $r = 8$, evaluate on a held-out set, and double if performance is insufficient.

**Alpha $\alpha$**: Set to $r$ by default (scaling factor = 1). Some practitioners fix $\alpha = 16$ and vary $r$, which means the scaling factor $\alpha/r$ changes with $r$. Others set $\alpha = 2r$ (scaling = 2), which can help on tasks requiring stronger adaptation. Don't over-tune this.

**Target modules**: Start with `q_proj` and `v_proj`. If performance is insufficient, add `k_proj` and `o_proj`. If still insufficient, add the MLP layers (`gate_proj`, `up_proj`, `down_proj` in LLaMA-style models).

**Learning rate**: LoRA generally benefits from higher learning rates than full fine-tuning. $10^{-4}$ to $3 \times 10^{-4}$ is a common range (vs. $10^{-5}$ to $5 \times 10^{-5}$ for full fine-tuning). The frozen base weights don't destabilize training, so you can be more aggressive.

### 9.6.3 HuggingFace PEFT: Code Walkthrough

The `peft` library is the standard implementation. Here's a complete example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",  # or any open-weight model; Llama 2 identifiers are gated/legacy
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # rank
    lora_alpha=16,                # alpha (scaling = alpha/r = 2)
    target_modules=["q_proj", "v_proj"],  # which weight matrices
    lora_dropout=0.05,            # dropout on LoRA layers
    bias="none",                  # don't train biases
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# "trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%"

# Training proceeds normally — only LoRA params receive gradients
trainer = Trainer(model=model, ...)
trainer.train()

# Save only the adapter (not the base model)
model.save_pretrained("./my_lora_adapter")
# Saves ~32 MB instead of 13 GB

# Later: load and merge
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
merged_model = model.merge_and_unload()  # merge BA into W, discard adapter
```

For QLoRA, add quantization config:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # double quantization
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",  # or any large open-weight model
    quantization_config=bnb_config,
    device_map="auto"
)
# Now 70B model fits in ~35 GB GPU memory
model = get_peft_model(model, lora_config)
```

### 9.6.4 Multiple LoRA Adapters: Task Routing

Because the base model is shared, you can maintain a library of LoRA adapters and swap them at inference time based on the incoming request. Think of it like feature flags or plugin architecture:

```python
from peft import PeftModel

base_model = load_base_model()

# Each adapter is ~30–100 MB
adapters = {
    "sql": PeftModel.from_pretrained(base_model, "./adapter_sql"),
    "medical": PeftModel.from_pretrained(base_model, "./adapter_medical"),
    "legal": PeftModel.from_pretrained(base_model, "./adapter_legal"),
}

def route_and_generate(request, task_type):
    model = adapters[task_type]
    return model.generate(request)
```

However, this naive approach requires loading and unloading adapter weights for each request, and the base model must be replicated in memory for each `PeftModel` instance (or handled carefully with shared references).

**S-LoRA** (Sheng et al., 2023) solves this properly: it batches requests across different LoRA adapters, stores all adapter weights in a unified GPU memory pool, and computes LoRA outputs for multiple adapters simultaneously using custom CUDA kernels. It can serve thousands of LoRA adapters on a single GPU cluster with near-zero per-adapter memory overhead and minimal latency impact.

---

## 9.7 Comparative Analysis

### 9.7.1 Performance vs. Parameter Efficiency

From the Xu et al. (2023) survey, aggregating across NLP benchmarks (GLUE, SuperGLUE, summarization, QA):

| Method | Trainable params (%) | Performance vs. full FT |
|---|---|---|
| Full fine-tuning | 100% | 100% (baseline) |
| LoRA ($r=8$) | 0.06–0.5% | 95–99% |
| Adapters ($m=64$) | 0.5–2% | 93–98% |
| Prefix Tuning | 0.1–1% | 88–95% |
| Prompt Tuning | <0.01% | 80–95% (model-size dependent) |
| BitFit | ~0.1% | 85–93% |

The "performance gap" narrows as model scale increases and as task complexity decreases. For instruction following with large models (>7B), LoRA at $r=8$ frequently matches full fine-tuning performance.

### 9.7.2 Training Speed

LoRA training is faster than full fine-tuning primarily due to smaller optimizer states, not faster forward/backward passes:

| Model | Full FT throughput | LoRA throughput | Speedup |
|---|---|---|---|
| LLaMA-7B | ~3,500 tok/s | ~4,200 tok/s | $1.2\times$ |
| LLaMA-13B | ~1,800 tok/s | ~2,400 tok/s | $1.3\times$ |
| LLaMA-65B (QLoRA) | N/A (OOM) | ~450 tok/s | Enables training |

The modest throughput speedup comes from fewer gradient computations (frozen weights don't need gradients) and smaller optimizer state updates. The bigger win is fitting larger models on fewer GPUs.

### 9.7.3 Method Selection by Task Type

Based on the survey's meta-analysis:

**Use LoRA when**: You have moderate to large amounts of task-specific data (>1K examples), the task requires learning new behaviors (not just style), you need zero-overhead inference, or you're adapting for generation tasks.

**Use prefix/prompt tuning when**: You have very limited data (<100 examples), the model is large (>10B), and the task is well-represented in the pre-training distribution (the model just needs to be "pointed" at the right behavior).

**Use adapters when**: You cannot merge at inference (e.g., you need adapter weights to remain separable for continual learning), and you're comfortable with the small latency overhead.

**The convergence result from Xu et al.**: Given sufficient adapter capacity, all methods approach full fine-tuning performance. The differences matter most in the low-capacity, low-data regime.

---

## 9.8 Connections to Broader Topics

### 9.8.1 LoRA and Stable Diffusion (Chapter 8)

LoRA is ubiquitous in diffusion model fine-tuning. The mechanics are identical: freeze the UNet weights, apply low-rank adapters to the attention layers (and sometimes the convolutional layers, treating them as $d \times d$ matrices). Training on 20–100 images of a specific subject or style produces a ~10 MB adapter that can be applied to any image generated by the base model.

The key difference from LLM LoRA: diffusion model LoRAs are more often used for style injection than task specialization, and they're commonly composed (add multiple LoRA adapters with different weights to blend styles). This composability — simply summing scaled $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ matrices — is a direct consequence of LoRA's additive structure.

### 9.8.2 PEFT and Vision-Language Models (Chapter 10)

VLMs (like LLaVA or InstructBLIP) require adapting both the language model backbone and the vision encoder, often with different datasets and objectives. LoRA enables targeted adaptation: you can fine-tune the language model's attention layers for instruction following while leaving the vision encoder frozen (or vice versa). This makes VLM fine-tuning feasible on consumer hardware and enables domain-specific VLMs (medical imaging, satellite imagery) without full fine-tuning budgets.

### 9.8.3 LoRA and Reinforcement Learning from Human Feedback

RLHF (the training paradigm behind ChatGPT and Claude) involves three steps: supervised fine-tuning, reward model training, and PPO/GRPO-based policy optimization. All three can use LoRA, but the policy optimization step particularly benefits: the reference policy (the SFT model) and the active policy (being updated by RL) must be compared via KL divergence. With LoRA, both policies share the frozen base weights; only the adapter differs. Computing the KL divergence reduces to comparing the LoRA outputs, which is far cheaper than running two full forward passes through separate model copies.

This is why modern RLHF pipelines (TRL, OpenRLHF) default to LoRA for the policy model. For readers interested in RL: the adapter-as-policy-delta interpretation is elegant — the LoRA adapter parameterizes how much the policy has moved from the reference policy, with the rank $r$ acting as a bottleneck on how far the policy can drift.

---

## 9.9 Summary

The key ideas from this chapter:

1. **Full fine-tuning doesn't scale** to multi-task deployment: 84 GB of GPU memory per training run, full model copies per task variant.

2. **PEFT methods** freeze the base model and train a small number of additional or reparameterized parameters, achieving 95–99% of full fine-tuning performance at 0.06–2% of the trainable parameter count.

3. **LoRA** is the dominant method: decompose $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ where $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$. Freeze $\mathbf{W}$, train $\mathbf{A}$ and $\mathbf{B}$ only. Initialize $\mathbf{B} = \mathbf{0}$ so training starts from pre-trained behavior.

4. **Zero inference overhead**: Merge $\mathbf{W}_{\text{merged}} = \mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ before serving. No adapter layers in the forward pass.

5. **Parameter reduction**: $256\times$ fewer optimizer states for a typical rank-8 LoRA on a 4096-dimensional weight matrix. Task-specific adapters are ~30–100 MB rather than ~14 GB.

6. **QLoRA** extends this to very large models: 4-bit quantization of the frozen base (NF4 + double quantization) enables training 65B models on a single GPU.

7. **Adapter libraries** (S-LoRA, vLLM) enable efficient serving of thousands of task-specific adapters on shared infrastructure — the systems architecture that makes PEFT practically deployable.

---

## Summary

- Full fine-tuning of large language models does not scale to multi-task deployment: a 7B-parameter model requires approximately 112 GB of GPU memory for training (weights, gradients, and optimizer states), and each task variant demands a full copy of model weights for serving.
- LoRA (Low-Rank Adaptation) exploits the empirical finding that fine-tuning weight updates $\Delta \mathbf{W}$ have low intrinsic rank. It decomposes the update as $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$, reducing trainable parameters by approximately $256\times$ for a typical rank-8 configuration on 4096-dimensional weight matrices.
- LoRA's initialization ($\mathbf{B} = \mathbf{0}$) ensures training begins from exactly the pre-trained behavior. At inference, the adapter merges into the base weights via $\mathbf{W}_{\text{merged}} = \mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$, producing zero latency overhead --- the deployed model has the identical compute graph as the original.
- QLoRA extends LoRA to very large models by quantizing the frozen base to 4-bit NormalFloat (NF4) with double quantization, enabling training of a 65B-parameter model on a single 80 GB GPU. The NF4 data type places quantization bins at the quantiles of a standard normal distribution, minimizing error for neural network weight distributions.
- Additive PEFT methods offer complementary tradeoffs: adapters insert bottleneck modules with nonzero inference overhead; prefix tuning prepends learnable key-value vectors into attention layers; prompt tuning prepends soft tokens to the input and approaches full fine-tuning performance as model scale increases beyond 10B parameters.
- Rank selection is empirical: $r = 4$ suffices for style and tone changes, $r = 8$ is a robust default, and $r = 16$--$64$ is appropriate for complex domain adaptation. Applying LoRA to query and value projections provides the best performance-to-parameter ratio; adding key, output, and MLP projections yields further gains at higher parameter cost.
- Adapter serving systems such as S-LoRA enable thousands of task-specific LoRA adapters to be served concurrently on shared infrastructure, with each adapter requiring only 30--100 MB of storage versus 14 GB for a full model copy.

---

## Key Equations Reference

| Name | Equation | Section |
|---|---|---|
| LoRA forward pass | $\mathbf{h} = \mathbf{W}\mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$ | 9.4.2 |
| LoRA low-rank decomposition | $\Delta\mathbf{W} \approx \mathbf{B}\mathbf{A},\; \mathbf{B} \in \mathbb{R}^{d \times r},\; \mathbf{A} \in \mathbb{R}^{r \times k}$ | 9.4.1 |
| LoRA merge at inference | $\mathbf{W}_{\text{merged}} = \mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ | 9.4.5 |
| Adapter bottleneck | $\text{Adapter}(\mathbf{h}) = \mathbf{h} + \mathbf{W}_{\text{up}} \cdot \sigma(\mathbf{W}_{\text{down}} \cdot \mathbf{h})$ | 9.3.1 |
| Prefix tuning (modified attention) | $\mathbf{K}' = [\mathbf{P}_K;\, \mathbf{K}],\; \mathbf{V}' = [\mathbf{P}_V;\, \mathbf{V}]$ | 9.3.2 |
| LoRA parameter count (per matrix) | $2dr$ vs. full $d \times k$ | 9.4.3 |
| DoRA weight decomposition | $\mathbf{W}' = (\mathbf{m} + \Delta\mathbf{m}) \cdot \frac{\mathbf{W} + \mathbf{B}\mathbf{A}}{\|\mathbf{W} + \mathbf{B}\mathbf{A}\|_c}$ | 9.5.3 |
| AdaLoRA SVD parameterization | $\Delta\mathbf{W} = \mathbf{P}\mathbf{\Lambda}\mathbf{Q}$ | 9.5.4 |

---

## Exercises

**Exercise 9.1** *(Computation)*

LLaMA-2-7B has 32 transformer layers. Each layer has four attention weight matrices — $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$, $\mathbf{W}_O$ — each of shape $4096 \times 4096$. You apply LoRA with rank $r$ to all four matrices in all 32 layers.

(a) Write a formula for the total number of LoRA trainable parameters as a function of $r$. Compute the value for $r \in \{4, 8, 16, 32, 64\}$.

(b) For $r = 8$, compute the parameter reduction ratio relative to full fine-tuning of only those attention weight matrices. Section 9.4.3 states a $256\times$ reduction for a single $d \times d$ matrix at $r = 8$, $d = 4096$ — verify this matches your formula.

(c) The HuggingFace PEFT code in Section 9.6.3 reports 4,194,304 trainable parameters for `r=8` targeting only `q_proj` and `v_proj`. Verify this number using $32 \times 2 \times 2 \times d \times r$ with $d = 4096$, $r = 8$. Why does this count use $2 \times d \times r$ per matrix rather than $d^2$?

(d) Now extend LoRA to also cover the three MLP projection matrices (`gate_proj`, `up_proj`, `down_proj`), each of shape $4096 \times 11008$ in LLaMA-2-7B. How many additional trainable parameters does this add at $r = 8$? What is the total trainable parameter percentage of the full 7B model?

**Exercise 9.2** *(Conceptual)*

The relationship between LoRA rank $r$ and task performance is empirical and task-dependent.

(a) Section 9.4.1 states that fine-tuning updates $\Delta\mathbf{W}$ have low intrinsic rank. Give an intuitive explanation grounded in the structure of the pre-training objective: why should adapting a model that already captures broad language structure require only a low-rank shift?

(b) Rank-order these three fine-tuning tasks from lowest to highest recommended $r$, and justify: (i) converting output style to a fixed JSON schema, (ii) teaching medical sub-specialty QA for a niche domain, (iii) teaching competition-level mathematics. Which task might also require LoRA on MLP layers in addition to attention, and why?

(c) A colleague trains LoRA at $r = 64$ and gets 98% of full fine-tuning performance; at $r = 8$ they get 96%. They conclude "always use $r = 64$." Identify two concrete costs of higher rank that this conclusion ignores, and describe a scenario where $r = 8$ is strictly preferable despite the 2% gap.

**Exercise 9.3** *(Computation)*

Compare GPU memory across four training configurations for LLaMA-2-7B. Use: fp16 = 2 bytes/param, fp32 = 4 bytes/param. AdamW stores two fp32 optimizer moments per trainable parameter.

(a) **Full fine-tuning**: all 7B parameters trainable. Compute memory for (i) fp16 forward-pass weights, (ii) fp32 master weight copy, (iii) fp16 gradients, (iv) two fp32 Adam moments. Sum to get total and compare to the 112 GB figure in Section 9.1.

(b) **LoRA $r=8$ on Q and V only**: frozen fp16 base model plus LoRA optimizer states. Trainable parameter count is 4,194,304 (from Exercise 9.1c). Compute total memory and the reduction factor versus full fine-tuning.

(c) **QLoRA $r=8$**: base model in NF4 (0.5 bytes/param, effectively). Compute base model memory in NF4 and add the LoRA optimizer states from (b). What total memory do you get, and at what model size (billions of parameters) does QLoRA become strictly necessary to fit on a single 80 GB A100?

(d) Explain why merging QLoRA adapters into the NF4 base model is not straightforward (unlike standard LoRA). What steps are needed to produce a merged, deployable model from a QLoRA checkpoint, and at which step might quality be lost?

**Exercise 9.4** *(Conceptual)*

Compare LoRA, adapter modules, prefix tuning, and prompt tuning across four dimensions.

(a) **Inference latency**: LoRA can be merged before serving for zero overhead; the others cannot. For each of the other three methods, explain precisely what extra computation they add to every forward pass, and estimate the relative overhead for a 7B model serving 1000 tokens/second.

(b) **Data regime**: The Xu et al. (2023) survey finds prompt tuning underperforms LoRA with abundant data (>10K examples) but nearly matches it for very large models with sparse data (<100 examples). Explain the mechanism: what does each method *learn* that is different, and why does model scale help prompt tuning more than it helps LoRA?

(c) **Composability**: LoRA adapters can be linearly combined: $\mathbf{W}_\text{merged} = \mathbf{W} + s_1\frac{\alpha}{r}\mathbf{B}_1\mathbf{A}_1 + s_2\frac{\alpha}{r}\mathbf{B}_2\mathbf{A}_2$. Describe one concrete use case for blending two LoRA adapters. Can prefix tuning or adapter modules be blended in the same way? Explain why or why not.

(d) **Continual learning**: A system must add a new task adapter every week without forgetting old tasks. Which PEFT method is best suited, and why? Explain what property of LoRA's inference-time merge specifically makes it *less* suitable for this use case than adapter modules.

**Exercise 9.5** *(Computation)*

QLoRA quantizes the frozen base model to NF4 with double quantization.

(a) A standard fp16 LLaMA-2-70B model requires $70 \times 10^9 \times 2$ bytes. Compute memory in GB. Then compute memory for NF4 (4 bits = 0.5 bytes/param). Double quantization saves approximately 0.5 additional bits per parameter — compute total memory including double quantization savings.

(b) The QLoRA paper reports a 65B model fits in approximately 35 GB. Verify consistency with your calculation from (a) applied to 65B parameters. What accounts for memory beyond model weights (name at least two components)?

(c) During QLoRA's forward pass, NF4 weights are dequantized to bf16 for the matrix multiply, then discarded. If an A100 delivers 312 TFLOPS for bf16 matmuls, and dequantization adds roughly 15% overhead, estimate the effective TFLOP throughput. Is this overhead worth paying for training? For serving at inference time?

(d) A team wants to deploy a QLoRA-trained model in production with minimal inference latency. Write the exact sequence of steps (from QLoRA checkpoint to serving), and identify the one step where information loss can occur.

**Exercise 9.6** *(Connection)*

Section 9.8.3 notes that LoRA is used in RLHF because computing the KL divergence between policy $\pi_\theta$ and reference policy $\pi_\text{ref}$ becomes cheap when both share the same frozen base.

(a) With LoRA, both $\pi_\theta$ and $\pi_\text{ref}$ use the same frozen base weights $\mathbf{W}$, with adapters $\Delta\mathbf{W}_\theta$ and $\Delta\mathbf{W}_\text{ref}$ respectively. The token-level KL is $\sum_t [\log \pi_\theta(y_t) - \log \pi_\text{ref}(y_t)]$. Explain why computing this requires only two lightweight adapter forward passes rather than two full $7B$-parameter forward passes. What is the computational saving for a $7B$ model with $r=8$ LoRA?

(b) The rank $r$ acts as a bottleneck on how far the policy can drift from the reference. If $r$ is too low, the policy cannot maximize reward sufficiently; if too high, it may reward-hack. Given the RLHF objective's KL penalty coefficient $\beta$ (Section 10.2), describe a principled approach to jointly selecting $r$ and $\beta$. What would you monitor during training to detect that $r$ is the binding constraint?

(c) Diffusion model LoRAs (Section 9.8.1) are routinely blended by summing $\Delta\mathbf{W}$ matrices with scalar weights to mix styles. Could the same technique work for LLM task adapters — e.g., blending a SQL-generation adapter with a medical-QA adapter? Predict what the blended model would do, and identify the fundamental limitation of this approach compared to multi-task fine-tuning.

---

## References

- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv:2106.09685*.
- Xu, L., et al. (2023). "Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment." *arXiv:2312.12148*.
- Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." *ICML 2019*.
- Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *ACL 2021*.
- Lester, B., et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." *EMNLP 2021*.
- Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
- Hayou, S., et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models." *arXiv:2402.12354*.
- Liu, S., et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." *arXiv:2402.09353*.
- Zhang, Q., et al. (2023). "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning." *ICLR 2023*.
- Aghajanyan, A., et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL 2021*.
- Sheng, Y., et al. (2023). "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." *arXiv:2311.03285*.
- Ben Zaken, E., et al. (2022). "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models." *ACL 2022*.
