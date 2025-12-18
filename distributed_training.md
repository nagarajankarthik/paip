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


Answer: ??

Perform the following experiments:

1. Try running the training with the `global_batch_size` parameter in the `train` section (i.e. `train.global_batch_size`) set to 16 and 64. In each case, adjust the `train.train_iters` parameter to ensure that the validation and test losses are below ?. How does the required total training time vary with the batch size? 

Answer: The required training time is inversely proportional to the batch size. The training takes a longer time as the batch size decreases because the number of optimizer steps increases.

2. Megatron-Bridge supports activation recomputation using the `recompute_granularity` parameter in the `model` section. The effect of activating this setting during training will be investigated in this task.

    a. Try running the training with `recompute_granularity` set to `full` and `selective`. This will require uncommenting the line containing `recompute_granularity` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. How do the `Step time` and `Throughput per GPU` metrics change as compared to the baseline run for which this setting was not enabled? Explain the reasons for the observed differences.

    b. Increase `model.seq_length` and `dataset.sequence_length` to 16384. Run the training without activation recomputation, and with `recompute_granularity` set to `full` and `selective`. How do the `Step time` and `Throughput per GPU` metrics compare for these three cases? If you encounter an error for any of these cases, explain the likely reasons. 

    c. Apart from the case of large sequence lengths, what are some other scenarios where activation recomputation may be useful?

Try training the model with and without activation recomputation. What is the effect on the training time and throughput? 






# References

* [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)

