# Distributed Training of Large Language Models

Modern Large Language Models (LLMs) are trained in parallel across multiple GPUs. The objective of this module is to explain the considerations relevant for training models at scale. 

# Challenges associated with LLM training

1. **Memory**: LLM training requires performing multiple forward and backward passes of numerous batches of data through a model containing numerous layers. During training, the model weights, gradients, activations and optimizer states need to be regularly accessed and updated. This can quickly overwhelm the video Random Access Memory (vRAM) available on a single GPU. The vRAM specification of a graphics card determines the maximum volume of data it can process at any given time.

2. **Throughput**: The need to process large volumes of data implies that the training cannot be completed within a reasonable time unless the throughput is sufficiently high. The batch size required for training at scale typically requires the training to be distributed across multiple GPUs, to enable high throughput without exceeding the memory available on a single GPU.


# Single GPU Training

## Theory

Read the section entitled "First Steps: Single GPU Training" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview).

Answer the following questions:


1. How does the batch size affect the overall training time?

2. What is activation recomputation? What is the key trade-off associated with it?

3. What is gradient accumulation? What is the relationship between the global and micro batch sizes?


## Practical

The hands-on exercises will involve training the Qwen3-4B model using the [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) framework. The training parameters can be configured using the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The existing set of parameters should be used as a baseline for comparison.


Perform an initial training run without changing any parameters in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. Take note of the following results from the training logs:

- Theoretical memory footprint (MB)
- Allocated and reserved memory after the first iteration
- Step time (s) 
- Throughput per GPU (TFLOP/s/GPU)
- Validation and test losses


---Answer Begin---
```
Theoretical memory footprints: weight and optimizer=69053.29 MB
[Rank 0] (after 1 iterations) memory (MB) | allocated: 69454.74462890625 | max allocated: 69454.76025390625 | reserved: 73472.0 | max reserved: 73472.0
Average Step Time: 4.097s
Average GPU Utilization: 200.33 TFLOP/s/GPU
```
---Answer End---

Perform the following experiments:

1. Try running the training with the `global_batch_size` parameter in the `train` section (i.e. `train.global_batch_size`) set to 16 and 64. How do the `Step time` and `Throughput per GPU` metrics vary with global batch size?


---Answer Begin---

| Global batch size | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| --- | --- | --- |
| 16 | 2.11 | 194.68 |
| 32 | 4.10 | 200.33 |
| 64 | 8.09 | 202.99 |


There is a slight increase in throughput with increasing global batch size because the optimizer step can be executed after peforming the forward and backward passes for more samples. Since the time required for the optimizer step depends only on the number of model parameters, the number of samples processed per second is slightly greater for larger global batch sizes.

The extent of increase in throughput decreases with increasing global batch size. Let $b$, $x$ and $y$ denote the global batch size, total time required for forward and backward passes per sample and time required for the optimizer step respectively. The number of samples processed per second is given by:

$$
Number of samples processed per second = \frac{b}{bx + y} = \frac{1}{x + \frac{y}{b}}
$$

As the global batch size becomes very large, the number of samples processed per second approaches the constant value of $1/x$. Hence, increasing global batch size does not lead to a significant increase in throughput when this quantity is already very large.

---Answer End---

2. Try running the training with the `micro_batch_size` parameter in the `train` section (i.e. `train.micro_batch_size`) set to 2, 4, 8 and 16. How does the throughput vary with micro batch size?

---Answer Begin---

| Micro batch size | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| --- | --- | --- |
| 1 | 4.10 | 200.33 |
| 2 | 2.25 | 364.83 |
| 4 | 2.06 | 398.66 |
| 8 | 1.96 | 419.49 |

__Answer End__






3. Megatron-Bridge supports activation recomputation using the `recompute_granularity` parameter in the `model` section. The effect of activating this setting during training will be investigated in this task.

    a. Try running the training with `recompute_granularity` set to `full` and `selective`. This will require uncommenting the line containing `recompute_granularity` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. How do the `Step time` and `Throughput per GPU` metrics change as compared to the baseline run for which this setting was not enabled? Explain the reasons for the observed differences.

    ---Answer Begin---

    | Setting | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
    | --- | --- | --- |
    | None | 4.10 | 200.33 |
    | Full | 6.67 | 122.97 |
    | Selective | 5.11 | 160.72 |

    Enabling full activation recomputation decreases the throughput considerably. In this case, none of the activations are stored during the forward pass, requiring them to be recomputed during the backward pass. As the activations account for a small proportion of the GPU memory usage for short sequence lengths, there is no benefit in enabling activation recomputation in such cases.

    The throughput is better for selective as compared to full recomputation as it only recomputes those activations which with a larger memory footprint and which are cheaper to recompute. See the [Megatron documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html) for more details.
    
    ---Answer End---



    b. Increase `model.seq_length` and `dataset.sequence_length` to 16384. Run the training without activation recomputation, and with `recompute_granularity` set to `full` and `selective`. How do the `Step time` and `Throughput per GPU` metrics compare for these three cases? If you encounter an error for any of these cases, explain the likely reasons. 

    ---Answer Begin---

    One encounters errors related to insufficient memory if activation recomputation is disabled or set to selective. For long sequence lengths, the activations account for a significant proportion of the vRAM usage and attempting to store even a subset of them may be infeasible. The training only works when activation recomputation is set to full. In this case, the average step time and throughput per GPU per second are 58.03 seconds and 349.04 TFLOP/s/GPU respectively.

    ---Answer End---

    c. Apart from the case of large sequence lengths, what are some other scenarios where activation recomputation may be useful?

    ---Answer Begin---
    It may be needed when training models with a large number of parameters as the hidden size would be large in such cases. It may also be required when using large batch size.

    ---Answer End---




# References

* [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)
* 
* [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/#pipelining-and-fsdp)
* [DeepSpeed Pipeline Paralleism](https://www.deepspeed.ai/tutorials/pipeline/)
* [Megatron Parallelism Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html#data-parallelism)
* [Megatron Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html#long-sequence-training)
