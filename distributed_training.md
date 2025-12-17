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








# References

* [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)

