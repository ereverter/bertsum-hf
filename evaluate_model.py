#!/usr/bin/env python
"""
Evaluate the model on the test set.
"""
# Set working directory
from pathlib import Path
import os
import sys
running_dir = Path(os.getcwd())
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# # Add relative directories to path
# sys.path.append('..')
# sys.path.append('../..')
# sys.path.append('../../..')

import argparse
from src.data_preparation import pad_sequence
from tqdm.auto import tqdm
from utils import seed_everything
from src.bertsum import BertSummarizer
from transformers import AutoTokenizer
import datasets
from functools import partial
import torch
import warnings
from copy import copy
import itertools
import evaluate
import numpy as np
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
rouge =  evaluate.load('rouge')

class CFG:
    pass

cfg = CFG()
cfg.seed = 3
seed_everything(cfg)

def preprocess_data(batch, tokenizer, max_src_ntokens=200, min_src_ntokens=5, max_nsents=100, min_nsents=3, max_length=512):
    # Get inputs keys
    inputs_keys = ['input_ids', 'labels', 'segment_ids', 'cls_ids', 'mask', 'mask_cls', 'src', 'tgt']
    batch_inputs = {key: [] for key in inputs_keys}

    # Get sentences
    src_list = batch['src']
    labels_list = batch['labels']
    tgt_list = batch['tgt']

    for src, labels, tgt in zip(src_list, labels_list, tgt_list):  # Each src is a list of sentences
        inputs = {key: [] for key in inputs_keys}

        # If data is empty, warn. Cannot skip as it would mess up the batch size
        if (len(src) == 0):
            warnings.warn('Empty data found')

        orig_src = copy(src)

        # Prepare the right input for BERT
        src = tokenizer(
            src,
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
        orig_src = [orig_src[i] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:max_nsents]
        labels = labels[:max_nsents]

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
        labels = labels[:len(cls_ids)]

        # Store data
        del _segs, segs, idxs
        inputs['input_ids'].append(src)
        inputs['labels'].append(labels)
        inputs['segment_ids'].append(segment_ids)
        inputs['cls_ids'].append(cls_ids)
        inputs['src'].append(orig_src)
        inputs['tgt'].append(tgt)

        # Append the preprocessed sample to the batch_inputs dictionary
        for key in inputs_keys:
            batch_inputs[key].extend(inputs[key])

    batch_inputs['input_ids'], batch_inputs['mask'] = pad_sequence(batch_inputs['input_ids'], padding_value=tokenizer.pad_token_id, max_length=None) # 512
    batch_inputs['labels'], _ = pad_sequence(batch_inputs['labels'], padding_value=-1, max_length=None) # cfg.max_nsents
    batch_inputs['segment_ids'], _ = pad_sequence(batch_inputs['segment_ids'], padding_value=0, max_length=None) # 512
    batch_inputs['cls_ids'], batch_inputs['mask_cls'] = pad_sequence(batch_inputs['cls_ids'], padding_value=0, max_length=None) # cfg.max_nsents

    return batch_inputs
    
@torch.no_grad()
def evaluate_model(model, data_test, device, rouge, topksent=3):
    keys_to_device = ['input_ids', 'segment_ids', 'cls_ids', 'mask', 'mask_cls']
    r1_list = []
    r2_list = []
    rL_list = []
    # rS_list = []
    for i in tqdm(range(len(data_test))):

        try:
            # Get logits
            model_inputs = {k: torch.tensor(v).to(device)[None, :] for k, v in data_test[i].items() if k in keys_to_device}
            outputs = model(**model_inputs)
            # # For bertsum
            # outputs = model(src=model_inputs['input_ids'], 
            #                 segs=model_inputs['segment_ids'], 
            #                 clss=model_inputs['cls_ids'], 
            #                 mask_src=model_inputs['mask'], 
            #                 mask_cls=model_inputs['mask_cls'])

            # Get predictions
            sent_indices = outputs['logits'].topk(topksent).indices.detach().cpu().numpy()[0]
            # sent_indices = outputs[0].topk(topksent).indices.detach().cpu().numpy()[0]

            summary = ' '.join(np.array(data_test[i]['src'])[sent_indices])
            reference = ' '.join(data_test[i]['tgt'])

            # Tokenize
            # summary = ' '.join(tokenizer.tokenize(summary))
            # reference = ' '.join(tokenizer.tokenize(reference))

            res = rouge.compute(predictions=[summary], references=[reference], use_stemmer=False, rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=True)
            # res = rouge.get_scores(summary, reference, avg=True)
            r1_list.append(res['rouge1'])
            r2_list.append(res['rouge2'])
            rL_list.append(res['rougeL'])
            # rS_list.append(res['rougeLsum'])

        except Exception as e:
            print(f'Error in example {i}. {e}')
            continue

    return r1_list, r2_list, rL_list#, rS_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use for evaluation. Checkpoint.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for evaluation.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length to use for evaluation.")
    parser.add_argument("--max_src_ntokens", type=int, default=200, help="Maximum number of tokens to use for evaluation.")
    parser.add_argument("--min_src_ntokens", type=int, default=5, help="Minimum number of tokens to use for evaluation.")
    parser.add_argument("--max_nsents", type=int, default=100, help="Maximum number of sentences to use for evaluation.")
    parser.add_argument("--min_nsents", type=int, default=3, help="Minimum number of sentences to use for evaluation.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results to.")
    parser.add_argument("--save_name", type=str, required=True, help="Name to save results to.")
    args = parser.parse_args()

    print('Trying to load from', running_dir / args.model)

    # Load dataset, model, and tokenizer
    dataset = datasets.load_from_disk(running_dir / args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(running_dir / args.model)
    model = BertSummarizer.from_pretrained(running_dir / args.model)
    # For bertsum
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # class aux_args:
    #     large = False
    #     temp_dir = None
    #     finetune_bert = False
    #     ext_ff_size = 2048
    #     ext_heads = 8
    #     ext_dropout = 0.2
    #     ext_layers = 2
    #     ext_hidden_size = 768
    #     max_pos = 512
    #     encoder = 'tr'

    # os.chdir('/home/usuaris/veu/enric.reverter/tfm/src')
    # sys.path.append('/models/bertsum')
    # sys.path.append('/models/bertsum/models')
    # print(running_dir / args.model)
    # print(os.getcwd())
    # checkpoint = torch.load(args.model)
    # model = ExtSummarizer(aux_args, device, checkpoint)
    # os.chdir(current_dir)

    model.to(device)
    model.eval()

    # Preprocess dataset
    preprocess_data_partial = partial(preprocess_data, 
                                      tokenizer=tokenizer, 
                                      max_length=args.max_length, 
                                      max_src_ntokens=args.max_src_ntokens,
                                      min_src_ntokens=args.min_src_ntokens,
                                      max_nsents=args.max_nsents,
                                      min_nsents=args.min_nsents)
    
    processed_data = dataset['test'].map(preprocess_data_partial, 
                                         batched=True, 
                                         batch_size=512, 
                                         load_from_cache_file=False, 
                                         desc='Preprocessing test set')
    
    # Evaluate model
    r1_list, r2_list, rL_list = evaluate_model(model, processed_data, device, rouge, topksent=3)
    results = {
        'r1': np.mean(r1_list),
        'r2': np.mean(r2_list),
        'rL': np.mean(rL_list),
        }
    
    print(results)

    # Save results
    with open(os.path.join(running_dir / args.save_dir, args.save_name), 'w') as f:
        json.dump(results, f)