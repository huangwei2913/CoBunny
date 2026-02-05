import os
import torch

from torch.utils.data import Sampler
from torch import nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, logger

from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


class BunnyTrainer(Trainer):

    def _get_train_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n" + "🚀"*10)
            print("DEBUG: BunnyTrainer._get_train_sampler 被触发了！")
            print(f"DEBUG: group_by_modality_length 状态: {self.args.group_by_modality_length}")
            print("🚀"*10 + "\n")

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        #dict：{"train_dataset": ..., "eval_dataset": ..., "data_collator": ...}。
        #在这里我们应该强制走这一个分支
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print("⚠️ 警告：没有启用长度分组，将使用默认采样器。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。")
            return super()._get_train_sampler(dataset)

    def create_optimizer(self):
        """
        Setup the optimizer with support for custom mm_projector/vision_tower learning rates.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # 确保 vision_tower 的解冻部分也能识别自定义学习率
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() 
                                      if "mm_projector" in name or "vision_tower" in name]
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
            # 1. 保存 Trainer 状态（Loss 曲线、Step 数等）
            super()._save_checkpoint(model, trial)

            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)

            # 2. 只有主进程执行合并与全量保存
            if self.args.local_rank <= 0:
                print(f"\n🚀 [Step {self.state.global_step}] 启动全量固化与配置同步...")

                import copy
                # 注意：在显存紧张的情况下，deepcopy 可能会导致 OOM。
                # 但为了保证 merge_and_unload 不影响训练中的梯度链，copy 是最稳妥的。
                temp_model = copy.deepcopy(self.model)
                
                # --- ✨ 核心逻辑：自动同步 ModelArguments 到 Config ---
                # 我们将 ModelArguments 中的所有字段同步到 config 中，确保保存后的 config.json 完整
                # 这样推理时 builder.py 才能正确识别你的 mixedencoder 结构
                
                attrs_to_sync = [
                    "model_type", "version", "freeze_backbone", "tune_mm_mlp_adapter",
                    "unfreeze_mm_vision_tower", "vision_tower", "unfreeze_vision_tower",
                    "use_s2", "mm_vision_select_layer", "pretrain_mm_mlp_adapter",
                    "mm_projector_type", "mm_resampler_type", "mm_use_im_start_end",
                    "mm_use_im_patch_token", "tune_mm_vision_resampler", "mm_mask_drop_mode",
                    "mm_mask_drop_skip_percentage", "mm_mask_drop_ratio", "mm_mask_drop_ratio_upper",
                    "mm_mask_drop_ratio_lower", "mm_vision_select_feature", "mm_dense_connector_type",
                    "vision_tower_dino", "vision_tower_siglip", "compression_K", "mm_hidden_size"
                ]

                for attr in attrs_to_sync:
                    # 优先从当前训练模型的 config 中取，取不到则保持 None
                    val = getattr(self.model.config, attr, None)
                    setattr(temp_model.config, attr, val)
                    
                # 标记这是一个已经固化过的模型，不再需要外挂 LoRA
                temp_model.config.lora_enable = False 

                # --- 3. 物理合并 LoRA (如果是 LoRA 模式) ---
                if getattr(self.args, 'lora_enable', False):
                    print("   🔗 正在执行权重合并 (Merge LoRA)...")
                    # merge_and_unload 会将线性层权重相加，返回原始结构的 Model
                    temp_model = temp_model.merge_and_unload()
                
                # --- 4. 执行全量物理保存 ---
                # save_pretrained 会自动根据权重体积生成 model.safetensors 或 pytorch_model.bin
                # 它会包含 LLM、Projector 和你解冻的 Vision Tower 全部 113+ 参数
                print(f"   💾 正在写入全量文件至 {output_dir} ...")
                temp_model.save_pretrained(output_dir)
                
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                
                # 显式清理内存
                del temp_model
                print(f"✅ [Checkpoint {self.state.global_step}] 保存成功！该目录可直接用于独立推理。")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if getattr(self.args, 'lora_enable', False):
            if self.args.local_rank <= 0:
                print(f"🚀 [全量化保存] 正在强制固化 Config 并合并权重...")
                
                import copy
                # 1. 深度拷贝模型
                temp_model = copy.deepcopy(self.model)
                
                # --- ✨ 修复点：安全地获取 ModelArguments 中的路径 ✨ ---
                # 注意：self.args 是 TrainingArguments，不含 vision_tower_dino
                # 我们需要使用 self.model.config 里的值，或者在初始化 Trainer 时传入的 model_args
                
                # 尝试从模型现有的 config 中提取，如果不存在，则使用默认逻辑
                # 通常在 train.py 初始化时，这些值已经被赋给了 model.config
                v_dino = getattr(self.model.config, 'vision_tower_dino', None)
                v_siglip = getattr(self.model.config, 'vision_tower_siglip', None)
                p_type = getattr(self.model.config, 'mm_projector_type', 'mlp2x_gelu')

                # 强制写入 temp_model 的 config，确保 save_pretrained 能写进 config.json
                temp_model.config.vision_tower_dino = v_dino
                temp_model.config.vision_tower_siglip = v_siglip
                temp_model.config.mm_projector_type = p_type
                temp_model.config.model_type = getattr(self.model.config, 'model_type', 'phi-1.5')
                # -------------------------------------------------------

                # 2. 物理合并 LoRA
                temp_model = temp_model.merge_and_unload()
                temp_model.config.lora_enable = False 
                
                # 3. 执行全量保存
                temp_model.save_pretrained(output_dir)
                
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                
                del temp_model
                print(f"✅ [成功] 全量模型及配置已保存至 {output_dir}")
        else:
            super(BunnyTrainer, self)._save(output_dir, state_dict)