#!/usr/bin/env python
'''
Fine-tunning HF Hub models for extractive summarization using the CNN dailymail dataset.
'''
import argparse
import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_PROJECT"]="bert-ext-cls"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

# Set working directory
running_dir = Path(os.getcwd())
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# # Add relative directories to path
# import sys
# sys.path.append('..')
# sys.path.append('../..')
# sys.path.append('../../..')

# Import libraries
import gc
from functools import partial

from transformers import AutoTokenizer, TrainingArguments
from datasets import load_from_disk
import torch
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.bertsum import BertSummarizer, BertSummarizerConfig
from utils import CFG, seed_everything, get_gpu_usage
from src.data_preparation import preprocess_train, preprocess_validation, SummarizerDataset, collate_batch
from src.trainer import SummarizerTrainer

from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune model for extractive summarization.')
    parser.add_argument('-i', '--input_data_path', type=str, default='cnn_dailymail', help='Path to the dataset.')
    parser.add_argument('-o', '--output_data_path', type=str, default='cnn_dailymail', help='Path to the dataset.')
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Config of the Hugging Face dataset.')
    parser.add_argument('-d', '--output_dir', type=str, default='processed_data', help='Output directory for the processed dataset.')

    args = parser.parse_args()

    print('Loading configuration...')
    cfg = CFG(running_dir / args.config_path, device)
    seed_everything(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)

    print('Loading dataset...')
    preprocess_train_partial = partial(preprocess_train, tokenizer=tokenizer, cfg=cfg)
    preprocess_validation_partial = partial(preprocess_validation, tokenizer=tokenizer, cfg=cfg)

    if cfg.needs_preprocessing:
        data = load_from_disk(running_dir / args.input_data_path)
        data_train = data['train'].map(preprocess_train_partial,
                            batched=True, 
                            batch_size=cfg.preprocessing_batch_size, 
                            load_from_cache_file=False,
                            remove_columns=data['train'].column_names, 
                            desc='Preprocessing train set')

        data_validation = data['validation'].map(preprocess_validation_partial,
                            batched=True, 
                            batch_size=cfg.preprocessing_batch_size, 
                            load_from_cache_file=False,
                            remove_columns=data['validation'].column_names, 
                            desc='Preprocessing validation set')
        
        data_test = data['test'].map(preprocess_validation_partial,
                            batched=True,
                            batch_size=cfg.preprocessing_batch_size,
                            load_from_cache_file=False,
                            remove_columns=data['test'].column_names,
                            desc='Preprocessing test set')
        
        if cfg.store_preprocessed_data:
            data['train'] = data_train
            data['validation'] = data_validation
            data['test'] = data_test

            data.save_to_disk(running_dir / args.output_data_path)
            del data
            gc.collect()
        
    else:
        data = load_from_disk(running_dir / args.input_data_path)
        data_train = data['train']
        data_validation = data['validation']

    # For debugging purposes
    if cfg.train_subset is not None:
        data_train = data_train.select(range(cfg.train_subset))

    if cfg.eval_subset is not None:
        data_validation = data_validation.select(range(cfg.eval_subset))
    
    print(get_gpu_usage())

    # Load custom extractive summarizer
    print('Loading model...')
    model_config = BertSummarizerConfig(checkpoint=cfg.checkpoint)
    model = BertSummarizer(config=model_config)

    print(get_gpu_usage())

    # Data loader
    print('Loading data...')
    train_dataset = SummarizerDataset(data_train)
    validation_dataset = SummarizerDataset(data_validation)

    print('Building trainer...')
    # Create a custom training argument class
    training_args = TrainingArguments(
        output_dir=running_dir/args.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        group_by_length=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.checkpoint_steps,
        save_steps=cfg.checkpoint_steps,
        include_inputs_for_metrics=True,
        prediction_loss_only=False,
        report_to="wandb",
        run_name=cfg.run_name,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
    )

    # Instantiate the custom trainer
    num_training_steps = cfg.num_epochs * len(train_dataset) / cfg.train_batch_size / cfg.gradient_accumulation_steps
    warmup_steps=int(cfg.warmup_ratio * num_training_steps)

    partial_collate_batch = partial(collate_batch, pad_token_id=tokenizer.pad_token_id)

    trainer = SummarizerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial_collate_batch,
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        device=cfg.device,
        cfg=cfg,
        use_pos_weight=cfg.use_pos_weight,
        pos_weight_alpha=cfg.pos_weight_alpha,
        )

    print('Fine-tuning model...')
    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(f"{running_dir / args.output_dir}/bertsum")

    wandb.finish()