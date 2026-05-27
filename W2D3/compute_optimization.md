# Strategies used to optimize memory utilization and throughput during distributed training

The Megatron framework provides users a number of options to optimize memory utilization and throughput that are applicable regardless of the specific parallelism configuration being used. The activations consume significant amounts of memory when training at long sequence lengths. Techniques to reduce the activation memory footprint include activation recomputation and offloading to CPU. Communication between GPUs introduces additional overhead that can reduce the throughput per GPU. Megatron includes functionality for overlapping communication with computation to enable efficient training, especially for large model sizes and long sequence lengths. The purpose of this note is to explain these techniques in greater detail. 

## Activation Memory

For each micro-batch, activations are initially calculated during the forward pass and are subsequently required during the backward pass. The simplest method to meet this requirement is to store the activations in GPU video Random Access Memory (vRAM) upon generation until the backward pass through the corresponding layer is complete. In this case, the memory requirements for the activations is [estimated](https://arxiv.org/pdf/2205.05198) to be

$$
\text{Total Activations Memory} = \frac{sbhL}{t}(34 + \frac{5as}{h})
$$

where $s$, $b$, $h$, $L$, $a$, and $t$ denote the sequence length, micro-batch size, hidden dimension size, number of transformer layers, number of attention heads, and number of attention heads respectively.

Since the second term scales as $s^2$, holding all the activations in vRAM may become infeasible at long sequence lengths. Activation recomputation and offloading to CPU are two strategies designed to overcome this problem.

### Activation Recomputation

This strategy trades off memory for compute by discarding a subset of the activations during the forward pass and recomputing them when required during the backward pass. 

The following are the key parameters used to control activation recomputation in Megatron-LM:

- `recompute_method`: Options are None (default), "uniform", "block". This parameter controls the number of layers for which activation recomputation is performed. The "uniform" option groups the transformer layers in a transformer block such that each unit contains `recompute_num_layers` layers. The input activation to the first layer of each group is always saved. The block option recomputes the activations for exactly `recompute_num_layers` layers per pipeline parallel stage, with the remaining activations bing saved during the forward pass.

- `recompute_num_layers`: Options are None (default) or an integer. If `recompute_method` is uniform, this parameter determines the number of layers per recomputation unit. If `recompute_method` is block, this parameter determines the number of layers per pipeline parallel stage for which activation recomputation is performed. Must be set to None if `recompute_method` is 'selective'.

- `recompute_granularity`: Options are None (default), "full", "selective". This option controls the extent of activation recomputation within a single transformer layer. The full option recomputes all the activations while the selective option only recomputes the activations for modules included in the `recompute_modules` list.

- `recompute_modules`: List of modules for which activation recomputation is performed. Defaults to "core_attn", meaning that recomputation is only performed for $QK^T$ matrix multiplication, softmax dropout and multiplication with $V$. This parameter is only used when `recompute_granularity` is set to "selective". 


For more details, see the [Megatron-LM](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/apidocs/core/core.transformer.transformer_config.html#core.transformer.transformer_config.TransformerConfig) and [Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/activation-recomputation.html) documentation.


