#!/usr/bin/env python
'''
Metrics for the fine tuning of the extractive summarization task.
'''
import numpy as np

# Evaluation functions
def extract_top_k_sentences(sentences, logits, k=3):
    if len(sentences) <= k:
        return " ".join(sentences)
    top_k_indices = logits.argsort()[-k:][::-1]
    top_k_sentences = [sentences[i] for i in top_k_indices]
    return " ".join(top_k_sentences)

def split_sentences(input_ids, tokenizer):
    sep_token = tokenizer.sep_token_id
    cls_token = tokenizer.cls_token_id
    pad_token = tokenizer.pad_token_id
    sentences = []
    sentence = []

    for token in input_ids:
        if token == sep_token:
            sentences.append(sentence)
            sentence = []
        elif token == cls_token:
            if sentence:
                sentences.append(sentence)
                sentence = []
        elif token == pad_token or token == -100:
            break
        else:
            sentence.append(token)

    if sentence:
        sentences.append(sentence)
        
    return [tokenizer.decode(s, skip_special_tokens=True) for s in sentences]

def evaluate_batch(model, logits, src_ids, tgt_ids, tokenizer, cfg):
    model.eval()

    # Prepare predictions and references
    predictions, references = [], []
    for i in range(len(src_ids)):
        original_sentences = split_sentences(src_ids[i], tokenizer)

        # Get top 3 sentences
        summary = extract_top_k_sentences(original_sentences, logits[i], k=3)
        predictions.append(summary)

        # Get reference summary

        reference = tokenizer.decode(tgt_ids[i][:np.where(tgt_ids[i] == 102)[0][0].item() + 1], skip_special_tokens=True)
        references.append(reference)

    # Compute Rouge scores
    scores = cfg.rouge.compute(predictions=predictions,
        references=references,
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_aggregator=True)

    return scores    