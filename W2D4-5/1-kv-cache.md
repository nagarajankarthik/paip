# Key Value Cache for Inference

## 1. Need for KV Cache during Inference


During **inference**, the model generates text **autoregressively** — one token at a time. Each newly generated token is appended to the sequence, and the model runs another forward pass to predict the next one. This means token positions are revealed sequentially, not all at once:

```
Step 1: ["Hello"] → predict "I"
Step 2: ["Hello", "I"] → predict "am"
Step 3: ["Hello", "I", "am"] → predict "a"
...
```

This sequential nature creates a key inefficiency: at every step, the model recomputes the key and value vectors for **all previous tokens**, even though those vectors haven't changed since the last step. KV cache is the solution to this redundancy.

## 2. Why is KV cache not used during training?


The main purpose of using a KV cache during inference is to avoid recomputing keys and values for previous tokens every time a new token is generated. During standard transformer training, the entire input sequence is available at once. The model computes queries, keys, and values for all tokens in a single batched forward pass, while causal masking ensures that each token attends only to earlier positions. Unlike autoregressive inference, there is no repeated recomputation of past tokens' K and V representations across decoding steps. Since training already computes each token's K and V exactly once per forward pass, the introduction of a KV increases GPU vRAM utilization without improving the throughput.

Training also requires backpropagation through the full computation graph. Intermediate activations, including K and V tensors, must be retained so gradients can be computed. These activations are already stored as part of the training graph, so maintaining a separate inference-style KV cache would typically add complexity without reducing computation or memory usage.

Therefore, conventional full-sequence training generally does not use KV caching. This is not because caching is fundamentally incompatible with gradient computation, but because the inference-time motivation for KV caching—avoiding repeated recomputation of past tokens—is absent during standard training. Specialized training methods may still use cached states, but they require carefully defined gradient handling and differ from the KV caches used during autoregressive inference.

## 3. Key considerations associated with the use of KV Cache

- **Substantial throughput improvement.** Without a KV cache, the key and value vectors for a token may be recomputed $O(N)$ times during generation of a sequence of length $N$. With KV caching, each token's key and value vectors are computed once and reused thereafter. Consequently, the time complexity of K/V projection is optimized from $O(N^2)$ to $O(N)$. The overall time complexity of autoregressively generating a sequence of length $N$ is still $O(N^2)$ because each new token attends to all previously generated tokens during the computation of the attention scores.

- **Increased memory usage.** The cache stores key and value tensors for every layer, every head, and every token in the context. For large models or long contexts, this can be significant. Memory grows as: `2 × n_layers × n_kv_heads × seq_len × head_dim × dtype_bytes`. The value of `dtype_bytes` is 2 for FP16 precision. Taking the [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) model as a representative example, the parameter values are `n_layers = 64`, `n_kv_heads = 8` and `head_dim` = 128. In this case, the cache size for a context length of 131,072 tokens is

$$
\text{KV cache memory} = \frac{2 \times 64 \times 8 \times 131,072 \times 128 \times 2}{10^9} \approx 34.4 GB
$$

This is comparable to the memory required for the model weights, which is around 64 GB. The use of Grouped Query Attention (GQA) limits the KV cache size. The [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) model has 64 attention heads but only 8 key-value heads. Without GQA, the KV cache size would be 8x larger, since the key and value vectors are no longer shared across attention heads.

- **Decoding becomes increasingly memory-bound.** Auto-regressive generation of each new token requires reading of model weights and the key and value vectors of all previous tokens from vRAM. As explained [here](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) and [here](https://medium.com/@arjunravi726/why-llm-inference-is-memory-bound-not-compute-bound-ba59c48739e0), the ratio of FLOPs to number of bytes transferred to and from memory is relatively low for this operation, making it memory-bound.

- **Specialized techniques are required to optimize KV cache management.** As explained in [this paper](https://arxiv.org/pdf/2603.20397), key classes of techniques used to optimize KV cache memory utilization include the following:

1. Cache eviction methods that selectively discard less critical tokens.
2. Compression and reconstruction techniques.
3. Hybrid memory solutions that make use of multi-tier storage.
4. State-of-the-art attention algorithms with custom methods for context processing.
5. Hybrid strategies that incorporate multiple optimization techniques.


## 4. How does KV cache work?

### The core idea

In the attention mechanism, keys (K) and values (V) for any token depend only on that token's embedding — not on any future tokens. This means that once K and V are computed for a token, they remain valid for all future generation steps. Only the query (Q) changes at each step, because the query represents "what the current token is looking for."

KV cache exploits this by saving K and V at the time they are first computed, and reusing them in all subsequent steps.

### Without KV cache

At step *n*, the full attention computation for a single seqeunce is:
1. Project all *n* tokens through W_K, W_Q, W_V → three matrices of shape `(n, d_out)`
2. Compute attention scores: Q @ K^T, shape `(n, n)`
3. Apply causal mask, softmax, multiply by V

The FLOPs for step 1 is proportional to `n`.

### With KV cache

At step *n*:
1. Load cached K and V for positions 0 to *n-1* — no computation needed
2. Compute K and V only for the new token — O(1) projection
3. Append new K, V to cache
4. Compute Q only for the new token — O(1) projection
5. Compute attention: Q_new @ K_all^T — only one query row, so the score matrix is shape `(1, n)` instead of `(n, n)`

The use of the KV cache optimizes the time complexity of Q, K and V projections from $O(n)$ to $O(1)$ at each decoding step.

See this [page](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) for more details.

## 5. Exercise

Modify the `gpt_ch04.py` script in this [repository](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/gpt_ch04.py) to include KV cache. Compare your results for inference throughput and memory utilization with the original script as well as the `gpt_with_kv_cache.py` script in the same repository. 

## 6. KV cache management techniques

### 6.1. Paged Attention 

A key challenge associated with serving a Large Language Model for online inference is the concurrent processing of multiple requests with variable input and output lengths that are unknown in advance. In such scenarios, efficient management of KV cache memory is essential for optimizing throughput, as measured by the number of tokens processed per second, and GPU utilization.

As explained [here](https://arxiv.org/pdf/2309.06180), a static allocation strategy in which the engine reserves a pre-specified amount of memory for the KV cache of each request results in significant memory wastage due to internal and external fragmentation. For this reason, modern inference frameworks such as [vLLM](https://github.com/vllm-project/vllm) use algorithms such as Paged Attention that enables storing the key and value vectors of the tokens within a sequence in non-contiguous memory. This design choice provides the following benefits:

- **Significant reduction of fragmentation.** KV-cache memory is allocated incrementally in fixed-size blocks as decoding progresses, eliminating the need to reserve space for the entire future sequence length in advance.
- **Maximizing reuse of previously allocated memory.** KV-cache blocks corresponding to identical prompt prefixes can be shared across multiple requests, avoiding duplication of KV-cache data and reducing overall memory consumption.

See this [page](https://www.aleksagordic.com/blog/vllm) for more details regarding KV cache memory management in vLLM.

### 6.2. Linear Attention

When performing inference using the standard attention mechanism, the time complexity of the attention calculation is $O(n^2d)$, where $n$ is the number of tokens in the input sequence and $d$ is the head dimension. The space complexity associated with the KV cache is $O(nd)$. The use of linear attention algorithms optimizes the time and space complexities to $O(nd^2)$ and $O(d^2)$ respectively. The key challenge is to do so without compromising the accuracy of the model's responses.

The most basic form of the linear attention algorithm evaluates the attention output vector for position $t$ as $o_t = \sum_{j=1}^t (q_t^Tk_j)v_j$, where $q$, $k$ and $v$ represent the query, key and value vectors and the subscript denotes the position along the sequence. The above update rule can be equivalently expressed using a state matrix $S_t$ as

$$
o_t = \sum_{j=1}^t v_j (k_j^T q_t) =  (\sum_{j=1}^t v_j k_j^T) q_t = S_t q_t
$$

The state matrix $S_t \in \mathbb{R}^{d \times d}$ can be viewed as an associative memory that maps keys to their corresponding values. As explained [here](https://sustcsonglin.github.io/blog/2024/deltanet-1/), the vanilla linear attention algorithm outlined above exhibits unsatisfactory performance primarily because it does not have a machanism for erasing irrelevant information obtained from earlier key-value pairs. 

As [Yang et al.](https://arxiv.org/abs/2412.06464v3) note, Mamba2 attempted to address this limitation by applying a position-dependent weight $\alpha_t \in (0, 1)$ to previous key-value pairs using the update rule:

$$
S_t = \alpha_t S_{t-1} + v_t k_t^T
$$

The key limitation of this approach is that it uniformly down-weights the contributions of all previous key-value pairs to the current state matrix $S_t$, without accounting for their relative importance. The [DeltaNet](https://sustcsonglin.github.io/blog/2024/deltanet-1/) algorithm was proposed as an alternative to remedy this deficiency. Its update rule is given by:

$$
S_t = S_{t - 1} - (S_{t - 1} k_t)k_t^T + (\beta_t v_t + (1 - \beta_t) S_{t - 1} k_t)k_t^T =  S_{t - 1} (1 - \beta_t k_t k_t^T) + \beta_t v_t k_t^T
$$

The first equality can be interpreted as dynamically erasing the old value $v_t^{old} = S_{t - 1}k_t $ associated with the current key and replacing it with a new value $v_t^{new} = \beta_t v_t + (1 - \beta_t)S_{t - 1}k_t$. As explained [here](https://sustcsonglin.github.io/blog/2024/deltanet-1/), the DeltaNet algorithm can also be interpreted as updating the state $S_t$ based on the prediction error associated with $S_{t - 1}$.

The Gated DeltaNet algorithm attempts to unify the advantages of Mamba2 and DeltaNet, resulting in the following update rule:

$$
S_t = \alpha_t S_{t - 1} (1 - \beta_t k_t k_t^T) + \beta_t v_t k_t^T
$$

As of June 2026, it is used in modern LLMs such as Qwen 3.6.
