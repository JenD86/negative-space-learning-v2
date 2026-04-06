from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",
    # TODO load from configs:
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# if training...

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
# prepare dataset too.

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        # TODO: pull from configs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_steps=5,
        num_train_epochs=3,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",  # based on paper!
        seed=42,
        output_dir="outputs",
        report_to="none",  # TODO: Setup wandb
    ),
)

# trainer.train() if training.

# if vllm export...

model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
