# Procedures to follow for exercises

The exercises in this section will require launching multiple training jobs using the [Nvidia Nemo Framework v25.11](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.11) container. The steps to launch a training job are as follows:

1. Launch an interactive Slurm job with 4 GPUs: `srun --gres=gpu:4 --pty bash`
2. Create enroot container using the provided sqsh archive: `enroot create -n test /mnt/weka/aisg/sqsh/nemo:25.11.sqsh`.
3. Launch the container: `enroot start --rw test`.
4. Activate the python environment: `source /opt/venv/bin/activate`.
5. Navigate to the working directory, which should be automatically mounted to the enroot container: `cd /mnt/weka/aisg/model_training_team/code_forge/yuli/repos/PAIP-Pretrain/W2D3`.
6. Create a `logs` folder in the working directory: `mkdir logs`.
7. Run the training job after setting the necessary parameters in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) configuration file. It is recommended to save the training logs to a file for future reference: `torchrun --nproc_per_node 2 qwen3_pretrain.py --config-file qwen3_pretrain_override.yaml | tee -a logs/test.log`.
8. Retrieve the throughput and memory usage metrics by running the `mean_flops.sh` bash script: `bash mean_flops.sh`.

Note that steps 1-6 only need to be performed once. 

Upon completing the exercises, exit both the container and the interactive Slurm job session. The `exit` command can be used in both cases.

# Distributed Training of Large Language Models

Modern Large Language Models (LLMs) are trained in parallel across multiple GPUs. The objective of this module is to explain the considerations relevant for training models at scale. 

# Challenges associated with LLM training

1. **Memory**: LLM training requires performing multiple forward and backward passes of numerous batches of data through a model containing numerous layers. During training, the model weights, gradients, activations and optimizer states need to be regularly accessed and updated. This can quickly overwhelm the video Random Access Memory (vRAM) available on a single GPU. The vRAM specification of a graphics card determines the maximum volume of data it can process at any given time.
2. **Throughput**: The need to process large volumes of data implies that the training cannot be completed within a reasonable time unless the throughput is sufficiently high. The batch size required for training at scale typically requires the training to be distributed across multiple GPUs, to enable high throughput without exceeding the memory available on a single GPU.

# Single GPU Training

## Theory

Read the section entitled "First Steps: Single GPU Training" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview).

Checklist of key concepts:

1. Effect of varying global batch size upon convergence and training time
1. Calculation of memory requirements for model parameters, gradients, activations and optimizer states
1. Activation recomputation and its associated trade-off
1. Gradient accumulation 
1. Relationship between the global batch size, micro batch size, and number of gradient accumulation steps

## Check your understanding

1. Which of the following statements regarding the GPU memory requirements for training are true? Select all that apply.

<ol type="a">
<li> The memory requirements for model parameters, gradients, and optimizer states increase with the model's vocabulary and hidden sizes. </li>
<li> The memory requirement for storing activations increases as the sequence length increases. </li>
<li> Memory utilization of model parameters, gradients and optimizer states increase when the batch size or sequence length is increased. </li>
<li> The memory footprint of the activations may become significantly larger than the model parameters, gradients or optimizer states if the sequence length is sufficiently large. </li>
</ol>

Answer: **a, b and d.**

- Statement a is **true** as the memory required for the model parameters, gradients and optimizer states are directly proportional to the number of parameters. Increasing the vocabulary size increases the size of the embedding and lm head layers. Increasing the hidden dimension increases the size of every layer in the model. 
- Statement b is **true**. In fact, the activation memory scales quadratically with sequence length, if the latter is sufficiently large.
- Statement c is **false**. The memory utilization of model parameters, gradients and optimizer states is independent of the batch size or sequence length. 
- Statement d is **true**. For long sequence lengths, activation memory scales quadratically with sequence length while the memory utilization of the other components remain constant. Hence, the activation eventually become the dominant component of memory usage.

2. Which of the following statements regarding activation recomputation are true? Select all that apply.

<ol type="a">

<li> Activation recomputation involves repeating some of the operations performed during the forward pass when backpropagating gradients. </li>
<li> Activation recomputation can be used to reduce the memory footprint of both activations and gradients. </li>
<li> Full activation recomputation reduces the activation memory footprint to a greater extent than selective activation recomputation. </li>

</ol>

Answer: **a and c.**

- Statement a is **true**. Activation recomputation discards some of the intermediate results of the forward pass and recomputes them subsequently during the backward pass.
- Statement b is **false**. Activation recomputation can only be used to reduce the memory footprint of activations but not the model parameters, gradients or optimizer states.
- Statement c is **true**. Full activation recomputation only saves activations at the transition point between successive layers of the transformer model, thereby requiring the forward pass to be repeated for each layer. In contrast, selective recomputation retains the results of feedforward computations, which reduces the volume of computation required during the backward pass at the expense of using more vRAM.

3. Which of the following statements regarding gradient accumulation are true? Select all that apply.

<ol type="a">

<li> Gradient accumulation may require multiple forward and backward passes of the same micro-batch through the model. </li>
<li> Gradient accumulation involves performing an optimizer step immediately after the forward and backward passes of each micro-batch, even if the number of gradient accumulation steps is greater than 1. </li>
<li> Gradient accumulation can be used in combination with activation recomputation to reduce the memory footprint associated with storing activations. </li>

</ol>

Answer: **c only.**

Explanation: 
- Statement a is **false**. Gradient accumulation does perform multiple forward and backward passes but each forward and backward pass processes a **different** micro-batch of data.
- Statement b is **false**. The optimizer step is performed only after gradients have been accumulated over all micro-batches.
- Statement c is **true**. Activation recomputation reduces memory footprint by recomputing some or all of the activations during the forward pass. Gradient accumulation reduces the number of sequences being processed during a single forward or backward pass. Both strategies reduce peak activation memory but do so in different ways.


## Practical

The hands-on exercises will involve training the Qwen3-4B model using the [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) framework. The training parameters can be configured using the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The existing set of parameters should be used as a baseline for comparison. All training runs in this section should be performed using a single H200 GPU.

Q1) Perform an initial training run without changing any parameters in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. Take note of the following results from the training logs:

- Step time (s) 
- Number of floating point operations per second per GPU (TFLOP/s/GPU)
- Allocated memory after the first iteration (GB)


Meaning of the metrics:

- Step time: Time taken to complete a single iteration of the training loop. This is the time required to perform `gradient_accumulation_steps` number of forward and backward passes, followed by an optimizer step in parallel across all ranks. Its key purpose is to estimate the time required to complete a training run given token and compute budgets.

- TFLOPs/s/GPU: This refers to the number of floating point operations (FLOPs) performed per second per GPU. As explained [here](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md), one can assess the extent to which the hardware is being used to its theoretical maximum capacity by comparing this quantity with the specifications provided by the manufacturer. For the Nvidia H200 GPUs used for this tutorial, the theoretical maximum TFLOPs/s/GPU attainable is [989](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#tflops-comparison-table). However, the theoretical maximum value is [never attained in practice](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#tflops-comparison-table), even if an application only performs matrix multiplications on matrices of the optimal dimensions without data transfer to and from the GPU. In a realistic LLM training run, there will be overhead associated with numerous operations such as disk I/O, communication and data transfer between the CPU and GPU as well as the GPU's High Bandwidth Memory (HBM) and Static Random Access Memory (SRAM). Consequently, the TLOPs/s/GPU measured in such settings will be significantly lower than the theoretical maximum. The Model and Hardware FLOPs Utilization (MFU and HFU) metrics report the estimated and actual TFLOPs/s/GPU attained as a percentage of the theoretical maximum. For an A100 GPU, the HFU ranges between [43.7 - 57.0%](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md) depending upon the model size.

For a standard transformer architecture, the number of FLOPs is calculated as follows in `Megatron-Bridge/src/megatron/bridge/training/utils/flop_utils.py`:


| Operation | FLOPs per training example excluding a constant pre-factor |
| --------- | ----- |
| query projection | `sequence_length` * `hidden_size` * `num_query_heads` * `head_dim` |
| key projection | `sequence_length` * `hidden_size` * `num_key_value_heads` * `head_dim` |
| value projection | `sequence_length` * `hidden_size` * `num_key_value_heads` * `head_dim` |
| query-key product | `sequence_length` * `head_dim` * `sequence_length` * `num_attention_heads` / 2 |
| softmax-value product | `sequence_length` * `sequence_length` * `head_dim` * `num_attention_heads` / 2 |
| post-attention projection |  `sequence_length` * `num_attention_heads` * `head_dim` * `hidden_size` |
| mlp projection | `sequence_length` * `hidden_size` * `mlp_proj_dim` |
| lm head projection | `sequence_length` * `hidden_size` * `vocab_size` |


`attention_flops_per_layer` = `query_projection` + `key_projection` + `value_projection` + `query_key_product` + `softmax_value_product` + `post_attention_projection`

`attention_flops` = `attention_flops_per_layer` * `num_attention_layers`

`mlp_flops` = `mlp_projection_flops` * `num_mlp_layers`

`total_flops` = `matmul_factor` * (`attention_flops` + `mlp_flops` + `lm_head_flops`)


Remarks on FLOPs calculation:

1. The calculation only includes the FLOPs for matrix multiplications. It does not include operations such as softmax, bias addition and layernorm, for which the number of FLOPs is several orders of magnitude smaller than matrix multiplications. As a representative example, consider an input tensor of dimensions $(B, L, D)$, where $B$, $L$ and $D$ are the batch size, sequence length and hidden size, respectively. Performing a forward pass through a linear layer with a weight matrix of dimensions $D \times D$ requires $B \times L \times D \times D = B.L.D^2$ FLOPs for the matrix multiplication, whereas the bias addition requires $B \times L \times D = B.L.D$ FLOPs. In addition, a layernorm operation along the hidden dimension requires $B \times L \times D = B.L.D$ FLOPs. It is seen that the FLOPs for the matrix multiplication is $D$ times as large as the FLOPs for the bias addition and layernorm. The hidden size, $D$, is typically $O(10^3)$ for a model that has tens of billions of parameters, which justifies the omission of operations whose time complexity is linear in $D$ in the FLOPs count.
2. The FLOPs calculation does not account for matrix multiplications performed within the optimizer step. Element-wise optimizers such as AdamW do not require matrix multiplications, while other optimizers like Muon do.  
3. The FLOPs for a single matrix multiplication operation involving a $m \times k$ matrix and a $k \times n$ matrix are $2 \times m \times k \times n$. The output matrix has dimensions $m \times n$. To calculate a single element of the output matrix, one must perform $k$ multiplications and $k$ additions. The FLOPs associated with various operations listed in the above table are calculated in this manner. 
4. The constant `matmul_factor` is typically set to 3 * 2 = 6. The factor of 2 arises for the reason explained in point 3. The factor of 3 accounts for the fact that gradient calculation in the backward pass will require two matrix multiplications for each such operation performed during the forward pass. For example, performing a matrix multiplication $C = A \cdot B$ in the forward pass requires calculating $\frac{\partial \mathbb{L}}{\partial A} = B \cdot \frac{\partial \mathbb{L}}{\partial C}$ and $ \frac{\partial \mathbb{L}}{\partial B} = A^T \cdot \frac{\partial \mathbb{L}}{\partial C}$. Note that a factor of 4 instead of 3 should be used if full activation recomputation is enabled, since each matrix multiplication required for the forward pass is performed twice.
5. For the query-key and softmax-value products, the FLOPs is divided by two to account for the causal mask. The use of the causal mask implies that only values in the lower triangular section of the resulting matrix are non-zero, thereby reducing the FLOPs by a factor of 2.

The `Step time` and `Throughput` metrics may exhibit slight variation with repeated attempts. Therefore, your results may not be identical to those shown below. The objective is to understand how the metrics vary as a function of the parameters in the [configuration file](qwen3_pretrain_override.yaml) rather than obtaining a specific number for each exercise.



A1)

Average Step Time: 3.37 s
Average GPU Utilization: 218.03 TFLOP/s/GPU
Allocated Memory : 73.65 GB

Perform the following experiments:


Q2) Try running the training with the `global_batch_size` parameter in the `train` section (i.e. `train.global_batch_size`) set to 16 and 64. How do the `Step time` and `Throughput per GPU` metrics vary with global batch size?

A2)

| Global batch size | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |  Memory per GPU (GB) |
| ----------------- | ------------- | -------------------------------- |  ------------------- |
| 16                | 1.78          | 230.78                           |   73.65              |
| 32                | 3.37          | 243.23                           |   73.65              |
| 64                | 6.57          | 249.96                           |   73.65              |




There is a slight increase in throughput with increasing global batch size because the optimizer step can be executed after peforming the forward and backward passes for more samples. Since the time required for the optimizer step depends only on the number of model parameters, the number of samples processed per second is slightly greater for larger global batch sizes.

The extent of increase in throughput decreases with increasing global batch size. Let $b$, $x$ and $y$ denote the global batch size, total time required for forward and backward passes per sample and time required for the optimizer step respectively. The number of samples processed per second is given by:

$$
Number of samples processed per second = \frac{b}{bx + y} = \frac{1}{x + \frac{y}{b}}
$$

As the global batch size becomes very large, the number of samples processed per second approaches the constant value of $1/x$. Hence, increasing global batch size does not lead to a significant increase in throughput when this quantity is already very large.

In practice, the global batch size is fixed at a relatively early stage of the hyperparameter tuning process primarily based on benchmark performance considerations. For a fixed value of global batch size and data parallelism degree, the micro batch size is set to the maximum value that can be accommodated by the memory available on a single GPU. The number of gradient accumulation steps is then automatically determined by the relation

$$
\text{global batch size} = \text{micro batch size} \times \text{number of gradient accumulation steps} \times \text{data parallelism degree}
$$

Q3) Try running the training with the `micro_batch_size` parameter in the `train` section (i.e. `train.micro_batch_size`) set to 2, 4, 8 and 16. How does the throughput vary with micro batch size? If you obtain an error for any of these cases, explain the likely reasons.

A3)

| Micro batch size | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| ---------------- | ------------- | -------------------------------- | ------------------- |
| 1                | 3.37          | 243.23                           | 73.65               |
| 2                | 2.14          | 382.64                           | 73.65               |
| 4                | 1.95          | 420.99                           | 73.62               |
| 8                | 1.86          | 441.32                           | 73.62               |


Increasing the micro batch size results in a considerable increase in throughput. The extent of increase is significantly greater than that obtained by increasing the global batch size. This is because tensors of larger size can be used to perform the computation in each forward and backward pass. Hence, the time required for each forward and backward pass increases less than proportionately to the extent of increase in batch size, which improves the training efficiency.

However, there is a limit to how large the micro batch size can be. For a micro batch size of 16, a CUDA out of memory error is encountered. In this case, the vRAM available on a single GPU is insufficient to store the activations for the entire batch.


Megatron-Bridge supports activation recomputation using the `recompute_granularity` parameter in the `model` section. The effect of activating this setting during training will be investigated in the next few questions.

Q4) Try running the training with `recompute_granularity` set to `full` and `selective`. This will require uncommenting the line containing `recompute_granularity` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. How do the `Step time` and `Throughput per GPU` metrics change as compared to the baseline run for which this setting was not enabled? Explain the reasons for the observed differences.
    
A4)

  | Setting   | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
  | --------- | ------------- | -------------------------------- | ------------------- |
  | None      | 3.37          | 243.23                           | 73.65               |
  | Full      | 5.53          | 148.29                           | 73.65               |
  | Selective | 4.30          | 190.56                           | 73.65               |

    Enabling full activation recomputation decreases the throughput considerably. In this case, none of the activations are stored during the forward pass, requiring them to be recomputed during the backward pass. As the activations account for a small proportion of the GPU memory usage for short sequence lengths, there is no benefit in enabling activation recomputation in such cases.
    The throughput is better for selective as compared to full recomputation as it only recomputes those activations which with a larger memory footprint and which are cheaper to recompute. See the [Megatron documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html) for more details.
    
Q5) Increase `model.seq_length` and `dataset.sequence_length` to 16384. Run the training without activation recomputation, and with `recompute_granularity` set to `full` and `selective`. How do the `Step time` and `Throughput per GPU` metrics compare for these three cases? If you encounter an error for any of these cases, explain the likely reasons. 
    
A5)    One encounters errors related to insufficient memory if activation recomputation is disabled or set to selective. The activation memory [scales quadratically with sequence length](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=memory_for_activations) for large sequence lengths. In this regime, increasing the sequence length from 1024 to 16384 results in the activation memory increasing by a factor of 256. At a sequence length of 16384, the [nanotron vRAM calculator](https://huggingface.co/spaces/nanotron/predict_memory) shows that the memory footprint for various components to be as follows for the Qwen3-4B model used in this tutorial:

| Component | Memory (GiB) |
| -------- | ------------ |
| BF16 model parameters | 6.8 |
| FP32 model parameters | 13.7 |
| FP32 gradients | 13.7 |
| Optimizer states | 27.3 |
| Activations | 85.6 |

When the sequence length is 16384, the vRAM required to store the activations for all layers becomes so large that it is infeasible to store even a subset of them. Hence, the training only works when activation recomputation is set to full. In this case, the average step time and throughput per GPU per second are 56.33 seconds and 359.53 TFLOP/s/GPU respectively.
    
Q6) Apart from the case of large sequence lengths, what are some other scenarios where activation recomputation may be useful?
    
A6)    It may be needed when training models with a large number of parameters as the hidden size would be large in such cases. It may also be required when using large batch size.
    
Q7) Try running the training for a sequence length of 16,384 without activation recomputation but with activation offloading to CPU enabled. What is the minimum number of layers that need to be offloaded in order for the training to work? 

A7)

The minimum number of layers is around 30. The average step time and GPU utilization are 53.92 s and 375.64 TFLOP/s/GPU respectively. It is slightly better than the throughput obtained using the full activation recomputation strategy. 


# Data Parallelism

A total of 4 H200 GPUs will be used for this task.

## Theory

Read the section entitled "Data Parallelism" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism).

Checklist of key concepts:

1. Strategies used to improve the efficiency of distributed data parallel (DDP) training
1. Relationship between global batch size, micro batch size, number of gradient accumulation steps and number of data parallel ranks
1. Limitations of using DDP with the model parameters, gradients and optimizer states replicated across all ranks
1. Effect of Zero-1 upon the efficiency of DDP and the type of communication operations performed during training
1. Differences between Zero-1, Zero-2 and Zero-3
1. Concept of prefetching
1. Key limitation of the ZeRO technique

## Check your understanding

1. Which of the following statements regarding DDP training are true? Select all that apply.

<ol type="a">
<li> Each rank holds its own copy of the model parameters, gradients and optimizer states. </li>
<li> A key benefit of DDP is the ability to process multiple micro-batches of data in parallel without exceeding the memory capacity of any single GPU. </li>
<li> For fixed values of micro-batch size and number of gradient accumulation steps, the global batch size is proportional to the number of data parallel ranks. </li>

</ol>

Answer: ** a, b and c.**

Explanation:
- Statement a is **true**. In DDP , the parameters, gradients and optimizer states are replicated across all ranks. The model parameters and optimizer states will always have identical values on all ranks. Each rank computes different gradients for the model parameters, since it processes its own shard of the global batch. Hence, the gradients are all-reduced on all ranks before the optimizer step.
- Statement b is **true**. If a particular combination of micro-batch size and sequence length can be accommodated within the memory constraints of a single GPU, DDP is a convenient method of increasing the global batch size and training throughput by processing multiple micro-batches in parallel. 
- Statement c is **true**. Increasing the number of data parallel ranks results in a proportional increase in the global batch size, assuming that the micro-batch size and number of gradient accumulation steps are fixed.

2. Which of the following statements regarding communication between DP ranks are true? Select all that apply. 

<ol type="a">

<li> The gradients are all-gathered on all ranks before the optimizer step. </li>
<li> The volume of data communicated between DP ranks scales with micro-batch size and sequence length. </li>
<li> Increasing the number of data parallel ranks increases the communication overhead associated with gradient synchronization across all ranks. </li>

</ol>


Answer: **c only.**

Explanation

- Statement a is **false**. The gradients are all-reduced on all ranks before the optimizer step.
- Statement b is **false**. The volume of data communicated between DP ranks scales with the number of model parameters but not the micro-batch size or sequence length.
- Statement c is **true**. Increasing the number of data parallel ranks reduces the throughput per GPU due to the increased communication overhead.

3. Which of the following statements regarding the Zero Redundancy Optimizer (ZeRO) are true? Assumethat the number of gradient accumulation steps is set to 1 so that the forward and backward passes are only performed for one micro-batch prior to the optimizer step. Select all that apply.

<ol type="a">
<li> Although ZeRO-1 and ZeRO-2 shard the optimizer states and gradients across all ranks, their communication cost is equivalent to that of DDP. </li>
<li> The sharding of model parameters implemented in ZeRO-3 necessitates an additional all-gather operation during the backward pass, which is not required for ZeRO-1 and ZeRO-2. </li>
<li> All the ZeRO variants only perform prefetching during the forward pass. </li>

</ol>


Answer: **a and b only.**

Explanation:

- Statement a is **true**. While DDP requires a single all-reduce operation to synchronize the gradients prior to the optimizer step, ZeRO-1 and ZeRO-2 perform an all-gather operation during the forward pass followed by a reduce-scatter operation during the backward pass. Since an all-reduce operation is equivalent to performing a reduce-scatter operation followed by an all-gather operation, the communication cost of ZeRO-1 and ZeRO-2 is equivalent to that of DDP.
- Statement b is **true**. Execution of the forward or backward pass through a layer requires all its parameters to be available on all ranks. Since the model parameters are sharded in ZeRO-3, they must be collected using an all-gather operation when required.
- Statement c is **false**. Prefetching overlaps the all-gather of model parameters with the computation for the forward or backward pass. ZeRO-3 requires prefetching during the backward pass.

4. When using ZeRO-3 with multiple gradient accumulation steps, should the reduce-scatter operation for gradients be performed during the backward pass for every micro-batch or only during the backward pass for the final micro-batch? 

Answer: The reduce-scatter operation is typically performed during the backward pass for each micro-batch. As explained [here](https://apxml.com/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/gradient-accumulation-sharding), the second scenario would require each GPU to store its local values of gradients for all model parameters. This is infeasible since ZeRO-3 is usually used when the memory footprint associated with the model parameters exceeds the capacity of a single GPU.

If there are multiple gradient accumulation steps, the all-reduce operation for DDP is overlapped with the backward pass for the last micro-batch. Similarly, the all-gather operation for ZeRO-1 and ZeRO-2 is overlapped with the forward pass for the first micro-batch, and the reduce-scatter operation is overlapped with the backward pass for the final micro-batch. 

## Practical

Q1) Run a DDP training job using the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file with the number of data parallel (DP) ranks set to 2 and 4. How does the throughput change with varying number of DP ranks?

A1)

Note that Megatron-LM internally calculates the number of DP ranks as `dp_size = world_size/(tp_size * pp_size * cp_size)` in `/megatron/core/parallel_state.py`. In this section, the number of DP ranks will always be equal to the world size since `tp_size = pp_size = cp_size = 1`. The `world_size` parameter can be adjusted using the `--nproc-per-node` flag in the torchrun command used to run the training.

For some reason, the Megatron-Bridge throughput is slightly different for the distributed and non-distributed optimizer cases even when running on a single GPU (i.e. world size = 1). All the results above are for the distributed optimizer case. The results for the single DP rank case in the following table were obtained for the non-distributed optimizer case.


| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| -------- | ------------- | -------------------------------- |
| 1        | 3.37          | 218.03                           |
| 2        | 2.05          | 199.92                           |
| 4        | 1.10          | 186.76                           |


The throughput per GPU decreases slightly with increasing number of DP ranks. This is because the communication overhead of the all-reduce operation used to synchronize gradients across devices increases as the number of DP ranks increases, since the total volume of data exchanged increases.



Q2) Repeat the 3 runs as question 1 using the same numbers of DP ranks but with the `overlap_grad_reduce` parameter set to `False` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. Does the throughput change as compared to the corresponding runs in question 1? Do not worry about trying to explain the reasons for your observations as they will be explored in the next question.

A2)

| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| -------- | ------------- | -------------------------------- |
| 1        | 3.90          | 210.23                           |
| 2        | 2.05          | 199.86                           |
| 4        | 1.10          | 186.07                           |


There is no significant change in the throughput when `overlap_grad_reduce` is set to `False`. The throughput per device still decreases to the same extent with increasing number of DP ranks. 

There are two reasons for this:

i. The communication of gradients cannot be completely overlapped with the backward pass since the gradients for the first few layers can only be all reduced after the backward pass is complete.

ii. Overlapping computation with communication results in concurrent execution of multiple kernels in different CUDA streams. As explained [here](https://anakli.inf.ethz.ch/papers/gpu_interference_socc25.pdf), this can lead to competition for GPU resources such as L2 cache, memory bandwidth, warp scheduling and compute pipelines.



Q3) Profiling is a useful tool to understand the effects of various parallelism configurations. Repeat the training run in question 1 for the 2 GPU case with Pytorch profiling enabled.

This requires setting the `use_pytorch_profiler` parameter to 'true' in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The profiling traces will be generated as files with the `.pt.trace.json` extension in the `nemo_experiments/default/tb_logs` subfolder of your working directory. Download one of the files and visualize them using the [Perfetto UI](ui.perfetto.dev).

When analyzing the traces, pay specific attention to the following points:

1. How many streams are present in the trace? What is the purpose of each stream? More information regarding CUDA streams can be found [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) and [here](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-semantics).

2. Which are the operations that account for the largest proportion of total wall duration in each stream? How many times were these operations executed? What is the average wall duration of these operations?

3. Do any of the operations identified in (b) take longer than average when overlapped with an operation from a different stream?

4. Are the executions of kernels on different streams perfectly overlapped? If not, what is preventing this from happening?

A3)


1. There are 2 streams in the trace. Stream 7 is for computation while stream 35 is for communication. In this case, communication occurs through the all-reduce operation.

2. In the computation stream, the `nvjet_tss_128x256_64x4_2x1_v_badd_coopA_NTN` kernel accounts for the largest proportion of total wall duration. It is executed 1152 times. The average wall duration of this kernel is 157.9 $\mu$s. In the communication stream, the `all_reduce` operation accounts for the largest proportion of total wall duration. It is executed 72 times. The average wall duration of this operation is 792.3 $\mu$s.

3. The `nvjet_tss_128x256_64x4_2x1_v_badd_coopA_NTN` kernel execution time takes significantly longer than average when overlapped with the nccl kernel responsible for the all-reduce operation. This is likely caused by the two kernels competing for the same resources on the GPU as explained [here](https://anakli.inf.ethz.ch/papers/gpu_interference_socc25.pdf).

4. The all-reduce operation is not entirely overlapped with the execution of the computation kernels. The gradients for the first few layers can only be synchronized after the backward pass is complete.



Q4) Use the findings from question 3 to explain why the throughput per device does not change significantly when the `overlap_grad_reduce` parameter is set to `false` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file.



The answer is stated in question 2.



Q5) Run the training on 1, 2 and 4 GPUs using FSDP with optimizer state sharding only. This requires setting the `use_megatron_fsdp` and `use_distributed_optimizer` parameters to 'true' in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The `data_parallel_sharding_strategy` parameter should be set to `optim`. The `ckpt_format` parameter should be set to `fsdp_dtensor`. Compare the throughput and memory usage for DDP and FSDP with optimizer state sharding. Run the PyTorch profiler for the FSDP case to obtain additional insights into the results. In this case, pay special attention to the profiling trace for the host (CPU) side. The traces for the various CPU threads are typically labelled using large integers such as `python 1789435`. You may also find it useful to go through [this paper](https://arxiv.org/abs/2304.11277), which explains the fundamentals of FSDP.

A5)


All results from this point onwards are obtained using the `nemo:25.11` container because neither the Megatron nor the Torch variants of FSDP work correctly with the `nemo:25.09.nemotron_nano_v2_vl` container.

Results for DDP with non-distributed optimizer using `nemo:25.11` container. The `Memory per GPU (GB)` column refers to the 'mem-allocated-gigabytes' field from the logs after one training iteration:


| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| -------- | ------------- | -------------------------------- | ------------------- |
| 1        | 4.10          | 200.11                           | 73.65               |
| 2        | 2.12          | 193.52                           | 73.65               |
| 4        | 1.13          | 181.03                           | 73.65               |


Results for FSDP:


| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| -------- | ------------- | -------------------------------- | ------------------- |
| 1        | 5.43          | 151.27                           | 72.60               |
| 2        | 2.77          | 147.96                           | 48.46               |
| 4        | 1.48          | 138.62                           | 36.39               |


The throughput is consistently smaller for FSDP as compared to DPP, even for the single GPU case. The PyTorch CPU profiling trace shows that the FSDP run invokes several hooks before and/or after the forward and backward passes through each FSDP unit. The need to process these hooks on the CPU side before and/or after every forward and backward pass through each FSDP unit increases the time interval between the launch of kernels that perform computation on the GPU, which reduces the throughput as compared to DDP. Two of the important functions performed by these hooks include: 

- Coordination of computation and communication operations. See section 4.3 of the [FSDP paper](https://arxiv.org/abs/2304.11277).
- Performing flatten and unflatten operations on parameters to facilitate computation using `FlatParameter` objects.

The key advantage of FSDP is the lower memory footprint per GPU as compared to DDP. For large models, this consideration becomes important.



Q6) Perform training using FSDP on 1, 2 and 4 GPUs for two additional cases. In the first case, shard the optimizer states and gradients by setting `data_parallel_sharding_strategy` to `optim_grads`. In the second case, include the model parameters in the sharding by setting this parameter to `optim_grads_params`. Compare the results for these two cases with those obtained in question 5.

A6)

Results for FSDP with optimizer state and gradient sharding:


| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| -------- | ------------- | -------------------------------- | ------------------- |
| 1        | 5.43          | 151.12                           | 72.60               |
| 2        | 2.78          | 147.78                           | 40.33               |
| 4        | 1.46          | 140.59                           | 24.25               |


Results for FSDP with optimizer state, gradients and model parameter sharding:


| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| -------- | ------------- | -------------------------------- | ------------------- |
| 1        | 5.74          | 142.86                           | 72.67               |
| 2        | 3.05          | 134.39                           | 38.22               |
| 4        | 1.55          | 132.60                           | 20.04               |


The memory footprint per GPU progressively decreases as the number of data parallel ranks increases and as more quantities are sharded. 

The throughput per GPU for the case in which both optimizer states and gradients are sharded is similar to the case in which only optimizer states are sharded. In comparison, the throughput per device decreases slightly if the model parameters are also sharded across data parallel ranks.

The sharding of model parameters requires all-gather operations to be performed during the forward and backward passes to materialize the parameters of the layer for which the computation is being performed. If the gradients are also sharded, reduce-scatter operations are also required to assign each DP rank its shard of the gradients. When the backward pass is overlapped with these communication operations, the resulting kernel interference increases the time required to perform the computation for the backward pass. This results in a slight decrease in throughput.



# Tensor Parallelism

## Theory

Read the section entitled "Tensor Parallelism" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism).

Checklist of key concepts:

1. Two main types of sharding strategies used in tensor parallelism (TP). 
1. Distributed communication operations used in each type of tensor parallelism
1. Optimal ordering of sharding strategies in a feedforward multi-layer perceptron (MLP) block
1. Trade-off between the memory savings and communication overhead arising from the use of tensor parallelism.
1. Throughput penalty associated with scaling TP across nodes
1. Use of sequence parallelism (SP) together with TP
1. Communication cost of using TP and SP together as compared with the cost of using TP alone

## Check your understanding

1. Which of the following statements regarding TP are true? Select all that apply.

<ol type="a">

<li>The key benefit of TP is the ability to shard the activations across multiple GPUs when training at long sequence lengths. </li>

<li> The communication overhead associated with the use of TP in a two layer MLP block is larger when using a column-linear followed by a row-linear sharding strategy, as compared to the opposite scenario. </li>

<li> The need to introduce communication operations along the critical path of computation limits the scalability of TP. </li> 


</ol>

Answer: ** a and c. **

Explanation:

- Statement a is **true**. For long sequence lengths, the activations is the component with the largest memory footprint. For a sufficiently large model, the use of FSDP may be insufficient to perform training even with a micro-batch size of 1, unless activation recomputation is enabled. TP can be a useful option in such situations.
- Statement b is **false**. The optimal strategy involves sharding the first layer using the column-linear strategy and the second layer using the row-linear strategy. This enables the forward pass to be performed for both layers using a single all-reduce operation. If the opposite strategy is used, the forward pass would require an all-reduce after the row-linear layer and an all-gather after the column-linar layer, thereby increasing the communication overhead. 
- Statement c is **true**. The need to introduce communication operations along the critical path of computation results in the throughput per GPU being sensitive to the speed of data transfer between them. Since the speed of data transfer is greater between GPUs on the same node as compared to GPUs on different nodes, a significant decrease in throughput is observed when TP is used across nodes.

2. Which of the following statements regarding sequence parallelism (SP) are true? Select all that apply.

<ol type="a">

<li> Both SP and TP are applied simultaneously to the same operations. </li>
<li> SP shards the activations along the sequence dimension while TP shards the weight matrices. </li>
<li> Combining TP and SP is equivalent to using TP alone in terms of communication overhead but offers additional memory savings. </li>

</ol>

Answer: ** b and c. **

Explanation:

- Statement a is **false**. SP and TP can be used in tandem but parallelize different operations. Specifically, SP is applied to the Dropout and LayerNorm operations while TP is applied to the self-attention and MLP operations.
- Statement b is **true**. Operations such as LayerNorm require access to the full hidden dimension for the purpose of calculating quantities such as mean and variance. Since these calculations can be performed independently for the various tokens comprising a sequence, SP is a natural choice for parallelizing such operations efficiently.
- Statement c is **true**. When using TP alone, all-reduce operations are required to obtain correct values of activations when exiting a TP region. In the TP + SP case, these all-reduce operations would be replaced by reduce-scatter operations. Hence, the final tensor saved in memory following the execution of a TP region would have dimensions [`batch_size`, `sequence_length`, `hidden_size`]. In the SP case, this decreases to [`batch_size`, `sequence_length`/`TP_degree`, `hidden_size`], thereby reducing the memory footprint of the activations by a factor of `TP_degree`.

## Practical

All results from this point onwards were obtained on SMC. The preceding results were obtained on Hopper.

Q1) Investigate the effect of setting the TP degree to 1, 2 and 4 with all other parameters set to their default values. The relevant parameter to be changed in [qwen3_pretrain_override.yaml](./qwen3_pretrain_override.yaml) is `tens or_model_parallel_size`. Ensure to set the world size to the same value as the TP degree. Explain the observed trends in throughput and memory utilization per GPU.


A1)

| TP Degree | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --------- | ------------- | -------------------------------- | ------------------- |
| 1         | 3.34          | 245.69                           | 73.65               |
| 2         | 3.84          | 106.72                           | 37.03               |
| 4         | 3.86          | 53.21                            | 18.61               |


The throughput per GPU decreases by at least 50% each time the TP degree is doubled. This is a direct consequence of the communication cost associated with TP, since the communication operations cannot be effectively overlapped with the forward and backward passes. 

The memory allocated per GPU decreases by approximately 50% each time the TP degree is doubled. This is because the model parameters and activations are sharded across increasing number of TP ranks. 



Q2) Repeat the runs in question 1 with sequence parallelism enabled for the cases TP=2 and 4 only. How do the throughput and memory utilization compare with the corresponding runs in question 1? Use the PyTorch profiler to explain the observed changes in the throughput between the TP only and TP + SP cases.

A2)


| TP Degree | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --------- | ------------- | -------------------------------- | ------------------- |
| 2         | 4.21          | 97.48                            | 37.03               |
| 4         | 4.34          | 47.22                            | 18.63               |


Enabling sequence parallelism reduces the throughput slightly as compared to using TP only. The PyTorch profiling trace shows that there is better overlap of the all-reduce communication operations with computation for the TP only case. When using TP in combination with SP, there are significant segments of the profiler trace where the communication blocks the computation, especially for the all-gather operation. 

The memory utilization remains unchanged since the activations only account for a small proportion of the VRAM utilization, since a short sequence length of 1024 was used.



Q3) Is it feasible to run the training on a single GPU for a long sequence length of 16384? How does the use of TP help in such cases?


A3)

One encounters a CUDA Out of Memory error if the training is performed on a single GPU. At long sequence lengths, the activations account for a significant proportion of the vRAM usage. Even sharding the activations across just two GPUs using TP = 2 is sufficient to make the training feasible.



# Context Parallelism

## Theory

Read the section entitled "Context Parallelism" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism).

Checklist of key concepts:

1. Difference between context parallelism (CP) and and tensor and sequence parallelism in combination.
2. Need for ring attention when using CP.
3. Use of zig-zag ring attention to distribute the computation load evenly across CP ranks.

## Check your understanding

1. Which of the following statements regarding context parallelism are true? Select all that apply.

<ol type="a">
<li>In regions of the model where TP is used, CP shards the activations along the sequence dimension. TP uses a similar mechanism with the only difference being that the sharding is performed along the hidden dimension.</li>
<li> CP does not require any communication for operations in which each token is processed independently of the others.</li>
<li> CP shards both model weights and activations.</li>
</ol>

Answer: **All statements are false.**

Explanation:
- Statement a is **false**. The mechanisms by which TP and CP operate are fundamentally different. CP shards the activations along the sequence dimension. TP shards the weights matrices but does not directly shard the activations. The distribution of activations along the hidden dimension across GPUs arises as a result of a forward pass through a linear layer sharded using the column-linear strategy. 
- Statement b is **false**. Although modules such as MLP and LayerNorm process each token independently of the others, an all-reduce operation is required to aggregate the gradients from all CP ranks.
- Statement c is **false**. CP shards the activations but not the model weights. The key difference between CP and SP is that CP shards the activations along the sequence dimension throughout every layer while SP performs this sharding only in regions where TP is not used.


2. Which of the following statements regarding ring attention are true? Select all that apply.

<ol type="a">
<li> Ring attention is needed when using CP because each GPU needs to access the key and value vectors for all tokens up to and including its shard of the sequence.</li>
<li> For fixed values of batch size and sequence length, the total volume of data transmitted per GPU during ring attention decreases as the CP degree increases.</li>
<li> For a fixed CP degree, the number of communication steps is proportional to the batch size and sequence length.</li>
</ol>


Answer: **a only.**

Explanation:
- Statement a is **true**. Using CP means that each GPU only holds the query, key and value tensors for its shard of the sequence at the beginning of the self-attention calculation. Ring attention is an efficient means of communicating the necessary key and value tensors to each GPU for the purpose of calculating the attention scores.
- Statement b is **false**. Ring attention involves $c_p - 1$ communication steps, where $c_p$ is the CP degree. During each step, the total volume of data transmitted per GPU is $s/c_p$, where $s$ is the total sequence length. Hence, the total volume of data transmitted per GPU during ring attention is $(s/c_p) * (c_p - 1)$. The factor $(c_p - 1)/c_p$ is a monotonically increasing function of $c_p$ which is bounded above by 1. Therefore, the total volume of data communicated increases with $c_p$. As $c_p$ increases, the rate of increase decreases.
- Statement c is **false**. The number of communication steps in ring attention is $c_p - 1$, which does not depend on batch size or sequence length.


## Practical

1. Perform a training run with the `context_parallel_size` parameter set to 2. Set all other parameters to their default values. Are there any memory savings as compared to the single GPU case?



The memory usage per GPU for CP = 2 is the same as the single GPU case. Unlike TP, CP only shards the activations but not the model weights. Since the default sequence length of 1024 is relatively small, the activations account for a very small proportion of the VRAM usage. Therefore, the memory utilization per GPU is the same as the single GPU case.



1. Repeat the run in question 1 with the sequence length increased to 16384. Compare the results obtained for throughput and memory utilization per GPU for the cases CP=2 and TP=2 with SP enabled for this sequence length.




| Case         | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| ------------ | ------------- | -------------------------------- | ------------------- |
| CP=2         | 23.60         | 429.08                           | 73.62               |
| TP=2 with SP | 24.29         | 416.83                           | 37.04               |


The throughput is slightly larger for CP = 2 as compared to TP = 2 with SP. As explained in the [Megatron-Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html), CP does not incur any additional communication overhead when applied to modules such as linear projection or normalization layers, as they do not perform inter-token operations. However, the application of TP to linear projection layers does require additional communication, which may not be completely overlapped with the computation for the forward and backward passes. 

On the other hand, the attention scores for each subset of heads in a multi-head attention layer can be calculated in parallel on different TP ranks without the need for additional communication. However, the use of CP in attention heads requires additional communication because of the need to calculate the dot product of the query vector at position $t$ with the key vectors at positions $1 \le t$. The net result is that the throughput is slightly higher for CP = 2 as compared to TP = 2 with SP. 

The memory utilization per GPU for the TP with SP case is about half that obtained for the CP = 2 case. While both TP and CP shard the activations, TP additionally shards the model parameters, gradients and optimizer states. This leads to the lower memory utilization per GPU for the TP = 2 with SP case.


