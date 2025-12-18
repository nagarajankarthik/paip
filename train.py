from megatron.bridge.recipes.qwen.qwen3 import qwen3_4b_pretrain_config
from megatron.bridge.recipes.qwen.qwen3 import qwen3_600m_pretrain_config
#from megatron.bridge.recipes.llama.llama32_1b import pretrain_config as llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

if __name__ == "__main__":
    cfg = qwen3_4b_pretrain_config(dir="./qwen3_4b_megatron", tensor_parallelism = 1,
                                   train_iters = 10, lr_warmup_iters=5, seq_length=1024)

    # cfg = qwen3_600m_pretrain_config(dir="./qwen3_600m_megatron", tensor_parallelism = 1,
    #                                train_iters = 10, lr_warmup_iters=5, seq_length=1024)
    # cfg = llama32_1b_pretrain_config(dir="./llama32_1b_megatron", tensor_parallelism = 1,
    #                                train_iters = 10, lr_warmup_iters=5, seq_length=1024)
    cfg.model.seq_length=1024
    cfg.logger.log_interval = 1
    cfg.checkpoint.load_optim = False
    # Override training parameters
    # cfg.train.train_iters = 10
    # cfg.scheduler.lr_decay_iters = 10000
    # cfg.model.vocab_size = 8192
    # cfg.tokenizer.vocab_size = cfg.model.vocab_size

    pretrain(cfg, forward_step)



