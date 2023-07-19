#!/usr/bin/env python
'''
Trainer for the fine tuning of the extractive summarization task.
'''
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

from transformers import Trainer, AdamW, EvalPrediction
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from src.metrics import evaluate_batch

class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, learning_rate, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.lr = learning_rate
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        warmup_factor = step * self.warmup_steps ** -1.5
        return [self.lr * min(step ** -0.5, warmup_factor) for _ in self.base_lrs]

class SummarizerTrainer(Trainer):
    def __init__(self, 
                 optimizer_class=None,
                 warmup_steps=None,
                 scheduler_class=None,
                 use_pos_weight=True, 
                 pos_weight_alpha=1.0,
                 device=None, 
                 cfg=None,
                 *args, 
                 **kwargs):
        super().__init__(*args, 
                         **kwargs,
                         compute_metrics=self.compute_metrics)
        self.optimizer_class = optimizer_class or AdamW
        self.warmup_steps = warmup_steps
        self.scheduler_class = scheduler_class or CustomLRScheduler
        self.use_pos_weight = use_pos_weight
        self.pos_weight_alpha = pos_weight_alpha
        self.device = device
        self.cfg = cfg

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        clean inputs so the default data collator does not add labels and attention mask
        """
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs["logits"]
        mask_cls = inputs["mask_cls"]
        mask_cls = mask_cls.float().view(-1)
        labels = inputs["labels"][:, :logits.shape[1]]

        pos_weight = None
        if self.use_pos_weight:
            pos_weight = self._get_pos_weight(labels)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        loss = loss_fct(logits.view(-1), labels.float().view(-1))
        loss = (loss * mask_cls).sum() / mask_cls.sum()

        return (loss, outputs) if return_outputs else loss
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Create the optimizer
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.learning_rate)

        # Create the custom learning rate scheduler
        custom_lr_scheduler = self.scheduler_class(
            optimizer,
            warmup_steps=self.warmup_steps,
            learning_rate=self.args.learning_rate,
            last_epoch=-1,
        )

        # Set the optimizer and scheduler
        self.optimizer = optimizer
        self.lr_scheduler = custom_lr_scheduler
    
    def _get_pos_weight(self, labels):
        pos = (labels == 1).sum()
        neg = (labels == 0).sum()

        if pos == 0:
            return torch.tensor(1.0)
            
        return max(neg / pos * self.pos_weight_alpha, torch.tensor(1.0))
    
    def prediction_step(self, model, inputs, prediction_loss_only, *args, **kwargs):
        model.eval()
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs["logits"]

            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return (loss.detach(), None, None)

            loss = self.compute_loss(model, inputs)
            logits = logits.detach()

        return (loss.detach(), logits, {"labels": inputs["labels"], "src_ids": inputs["src_ids"], "tgt_ids": inputs["tgt_ids"]})

    def compute_metrics(self, p: EvalPrediction):
        logits = p.predictions
        src_ids = p.label_ids["src_ids"]
        tgt_ids = p.label_ids["tgt_ids"]

        # Use evaluate_batch to compute Rouge scores
        scores = evaluate_batch(self.model, logits, src_ids, tgt_ids, self.tokenizer, self.cfg)

        return {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
        }