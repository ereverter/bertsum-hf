#!/usr/bin/env python
'''
Auxiliary functions for the fine tuning of the extractive summarization task.
'''
import json
import evaluate
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import subprocess
import random
import os
import numpy as np
import itertools

# Configuration details for training #
class CFG:
    def __init__(self, config_file, device):
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # General configuration
        self.checkpoint = config["checkpoint"]
        self.seed = config["seed"]    

        # Preprocessing configuration
        self.needs_preprocessing = config["needs_preprocessing"]
        self.store_preprocessed_data = config["store_preprocessed_data"]
        self.preprocessing_batch_size = config["preprocessing_batch_size"]

        self.summary_size = config["summary_size"]
        self.min_src_ntokens = config["min_src_ntokens"]
        self.max_src_ntokens = config["max_src_ntokens"]
        self.min_nsents = config["min_nsents"]
        self.max_nsents = config["max_nsents"]

        # Model configuration
        self.head_type = config["head_type"]

        # Training configuration
        self.num_workers = config["num_workers"]
        self.train_batch_size = config["train_batch_size"]
        self.eval_batch_size = config["eval_batch_size"]
        self.train_subset = config["train_subset"]
        self.eval_subset = config["eval_subset"]
        self.warmup_ratio = config["warmup_ratio"]
        self.lr = config["lr"]
        self.num_epochs = config["num_epochs"]
        self.gradient_accumulation_steps = config["gradient_accumulation_steps"]

        self.run_name = config["run_name"]
        self.checkpoint_steps = config["checkpoint_steps"]
        self.logging_steps = config["logging_steps"]

        self.use_pos_weight = config["use_pos_weight"]
        self.pos_weight_alpha = config["pos_weight_alpha"]
        self.device = device

        # Evaluation configuration
        self.rouge = evaluate.load('rouge')

# Auxiliary functions for preprocessing the data #
def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def lcs_length(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    return lengths[-1][-1]

def cal_rouge_l(evaluated_tokens, reference_tokens):
    reference_count = len(reference_tokens)
    evaluated_count = len(evaluated_tokens)

    lcs = lcs_length(evaluated_tokens, reference_tokens)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = lcs / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = lcs / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

# Most basic tokenization possible #
def tokenize_text_to_sentences(text):
    return sent_tokenize(text)

def tokenize_sentences_to_words(sentences, lower=True):
    return [word_tokenize(sentence.lower()) for sentence in sentences]

# Auxiliary functions for processes #
def get_gpu_usage():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader,nounits'
        ])
    
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split()]
    print(f'GPU: {gpu_memory[0]/1024**3}\nCUDA: {torch.cuda.memory_allocated()/1024**3}')

def seed_everything(cfg):
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

# Auxiliary functions for inference #
def prepare_sample(sample,
                   tokenizer,
                   max_src_ntokens=200, 
                   min_src_ntokens=5,
                   max_nsents=100,
                   max_length=512,
                   return_tensors=True):
    """
    Prepare sample to run inference.
    `sample` is of the form [sentence1, sentence2, ...]
    """
    inputs = {}

    # Prepare the right input for BERT
    src = tokenizer(
        sample,
        max_length=max_src_ntokens,
        truncation=True,
        stride=0,
        return_token_type_ids=False,
        return_attention_mask=False
    )
    
    # Ignore senteces that are too short
    # *Assumption*: if sentence is short it is not relevant
    idxs = [i for i, sentence in enumerate(src['input_ids']) if (len(sentence) > min_src_ntokens)]

    # Trim sentences to a maximum. Note they are already trimmed by the tokenizer
    src = [src['input_ids'][i] for i in idxs]
    sample = [sample[i] for i in idxs]
    src = src[:max_nsents]

    # Flatten into a single sequence (sents will be separated by [SEP] and [CLS] tokens already)
    src = list(itertools.chain(*src))
    if len(src) > max_length:
        src = src[:max_length-1] + [tokenizer.sep_token_id] # Truncate to 512 tokens

    # Intercalate 0s and 1s to differentiate between sentences
    _segs = [-1] + [i for i, t in enumerate(src) if t == tokenizer.sep_token_id]
    segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
    segment_ids = []
    for i, s in enumerate(segs):
        if (i % 2 == 0):
            segment_ids += s * [0]
        else:
            segment_ids += s * [1]

    # Get [CLS] positions, trim labels
    cls_ids = [i for i, t in enumerate(src) if t == tokenizer.cls_token_id]
    sample = sample[:len(cls_ids)]

    # Store data
    del _segs, segs, idxs
    inputs['input_ids'] = torch.tensor(src).unsqueeze(0) if return_tensors else src
    inputs['mask'] = torch.tensor([1] * len(src)).unsqueeze(0) if return_tensors else [1] * len(src)
    inputs['segment_ids'] = torch.tensor(segment_ids).unsqueeze(0) if return_tensors else segment_ids
    inputs['cls_ids'] = torch.tensor(cls_ids).unsqueeze(0) if return_tensors else cls_ids
    inputs['mask_cls'] = torch.tensor([1] * len(cls_ids)).unsqueeze(0) if return_tensors else [1] * len(cls_ids)
    inputs['sample'] = sample

    return inputs