# Strategies used to optimize memory utilization and throughput during distributed training

The Megatron framework provides users a number of options to optimize memory utilization and throughput that are applicable regardless of the specific parallelism configuration being used. The activations consume significant amounts of memory when training at long sequence lengths. Techniques to reduce the activation memory footprint include activation recomputation and offloading to CPU. Communication between GPUs introduces additional overhead that can reduce the throughput per GPU. Megatron includes functionality for overlapping communication with computation to enable efficient training, especially for large model sizes and long sequence lengths. The purpose of this note is to explain these techniques in greater detail. 

## Activation Memory

For each micro-batch, activations are initially calculated during the forward pass and are subsequently required during the backward pass. The simplest method to meet this requirement is to store the activations in GPU video Random Access Memory (vRAM) upon generation until the backward pass through the corresponding layer is complete. In this case, the memory requirements for the activations is [estimated](https://arxiv.org/pdf/2205.05198) to be

$$
\text{Total Activations Memory} = \frac{sbhL}{t}(34 + \frac{5as}{h})
$$

where $s$, $b$, $h$, $L$, $a$, and $t$ denote the sequence length, micro-batch size, hidden dimension size, number of transformer layers, number of attention heads, and tensor parallel size respectively.

Since the second term scales as $s^2$, holding all the activations in vRAM may become infeasible at long sequence lengths. Activation recomputation and offloading to CPU are two strategies designed to overcome this problem.

### Activation Recomputation

This strategy trades off memory for compute by discarding a subset of the activations during the forward pass and recomputing them when required during the backward pass. 

The following are the key parameters used to control activation recomputation in Megatron-LM:

- `recompute_method`: Options are None (default), "uniform", "block". This parameter controls the number of layers for which activation recomputation is performed. The "uniform" option groups the transformer layers in a transformer block such that each unit contains `recompute_num_layers` layers. The input activation to the first layer of each group is always saved. The block option recomputes the activations for exactly `recompute_num_layers` layers per pipeline parallel stage, with the remaining activations bing saved during the forward pass.

- `recompute_num_layers`: Options are None (default) or an integer. If `recompute_method` is uniform, this parameter determines the number of layers per recomputation unit. If `recompute_method` is block, this parameter determines the number of layers per pipeline parallel stage for which activation recomputation is performed. Must be set to None if `recompute_method` is 'selective'.

- `recompute_granularity`: Options are None (default), "full", "selective". This option controls the extent of activation recomputation within a single transformer layer. The full option recomputes all the activations while the selective option only recomputes the activations for modules included in the `recompute_modules` list.

- `recompute_modules`: List of modules for which activation recomputation is performed. This parameter is only used when `recompute_granularity` is set to "selective".

Defaults to "core_attn", meaning that recomputation is only performed for core attention operations such as $QK^T$ matrix multiplication, softmax, softmax dropout and multiplication with $V$. As explained on page 9 in Section 5 of [this paper](https://arxiv.org/pdf/2205.05198), this choice enables significant activation memory savings with minimal additional FLOPs used for recomputation. This can be better understood by analyzing the dimensions of the input tensor and the number of FLOPs required for each operation. The batch dimension is irrelevant for this analysis and has been omitted from the table below.

| Operation | Input Tensor Dimensions | FLOPs | Ratio of FLOPs to number of input elements |
| --- | --- | --- | --- |
| $QK^T$ | $sd$ | $ds^2$ | $s$ |
| Softmax | $s^2$ | $s^2$ | $1$ |
| Softmax Dropout | $s^2$ | $s^2$ | $1$ |
| $QK^T \times V$ | $s^2$ for $QK^T$, $sd$ for $V$ | $s^2d$ | $d$, assuming $d << s$ |
| $Q$ projection | $sh$ | $shad$ | $ad$ |
| $KV projection$ | $sh$ | $sha_{kv}d$ | $a_{kv}d$ |
| MLP up-projection | $sh$ | $sch^2$, where $c$ is a constant integer | $ch$ |

It is seen that the ratio of FLOPs to the number of input elements is significantly smaller for the softmax, dropout and multiplication with $V$ operations as compared to other operations. These operations also have larger numbers of input elements, especially if $s$ is chosen to be greater than $h$. Therefore, selectively recomputing the activations for the core attention operations enables significant memory savings with minimal additional compute overhead incurred during the backward pass. 


For more details, see the [Megatron-LM](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/apidocs/core/core.transformer.transformer_config.html#core.transformer.transformer_config.TransformerConfig) and [Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/activation-recomputation.html) documentation.

## Offloading Activations to CPU

It is also possible to offload the activations to CPU memory and reload them when required. This approach can enable the same extent of memory savings as activation recomputation without the additional computation overhead associated with repeating some of the operations performed during the forward pass. However, there will be additional overhead associated with data transfer between the GPU and CPU. Instructions for enabling this option can be found in the [Megatron-Bridge documentation](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/cpu-offloading.html).


## Overlapping Communication with Computation

Overlapping communication with computation is a general strategy that can be used to improve throughput by executing these operations in parallel. Megatron typically achieves this by performing communiation and computation in separate CUDA streams. A CUDA stream is a sequence of operations that are executed in order on the GPU and independently of other streams. Operations in different streams can be executed in parallel, enabling efficient utilization of the GPU's compute resources. Streams can be synchronized using events, which can be useful when one rank needs data from other ranks in order to perform an operation. 


The specific manner in which computation and communication are overlapped depends on the types of parallelism being used. For data parallelism, the two key techniques for achieving this include overlapping the all-gather of model parameters with the forward pass and overlapping the all-reduce or reduce-scatter of gradients with the backward pass. These optimizations can be enabled by setting the `overlap_grad_reduce` and `overlap_param_gather` options to `True` in the Megatron training configuration.


When tensor parallelism (TP) is combined with sequence parallelism (SP), all-gather and reduce-scatter operations are needed when transiting between model regions using different parallelism strategies. These coomunication operations can be partially overlapped with computation using a minimal set of configuration parameters in Megatron-Bridge. Note that sequence parallelism must be enabled to enable overlapping.


For context parallelism (CP), the ring attention mechanism is where most of the communication occurs. As explained [here](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zig-zag_ring_attention_%E2%80%93_a_balanced_compute_implementation), each rank can pre-fetch the next key-value pair while calculating the attention scores corresponding to its current key-value pair. This ensures that no rank is waiting for data from its neighbour in order to calculate the attention scores. No additional configuration is needed to enable this overlapping strategy. 

For Mixture-of-Experts (MoE) models, Megatron enables overlapping MoE expert parallel all-to-all communication with computation. See the [Megatron-Bridge documentation](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/communication-overlap.html#moe-expert-parallel-communication-overlap) for more details.

