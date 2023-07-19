#!/usr/bin/env python
'''
Data preparation for the fine tuning of the extractive summarization task.
'''
import itertools
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
import warnings

# Dataset mapping #
def preprocess_train(batch, tokenizer, cfg):
    """
    Simple preprocessing of the data for the extractive summarization task training.
    Assumes that the data is a Dataset with the keys: 'src' and 'labels'.
    `cfg` is a configuration object with the following attributes:
        - max_src_ntokens: maximum number of tokens in the source text
        - min_src_ntokens: minimum number of tokens in the source text
        - max_nsents: maximum number of sentences in the source text
        - min_nsents: minimum number of sentences in the source text
    Returns the preprocessed batch ready for the model.
    """
    # Get inputs keys
    inputs_keys = ['input_ids', 'segment_ids', 'cls_ids', 'mask', 'mask_cls', 'labels']
    batch_inputs = {key: [] for key in inputs_keys}

    # Get sentences
    src_list = batch['src']
    labels_list = batch['labels']

    for src, labels in zip(src_list, labels_list): # Each src is a list of sentences
        inputs = {key: [] for key in inputs_keys}

        # If data is empty, warn. Cannot skip as it would mess up the batch size
        if (len(src) == 0):
            warnings.warn('Empty data found')

        # Tokenize the list of sentences
        src = tokenizer(
            src,
            max_length=cfg.max_src_ntokens,
            truncation=True,
            stride=0,
            return_token_type_ids=False,
            return_attention_mask=False
        )

        # Ignore senteces that are too short
        idxs = [i for i, sentence in enumerate(src['input_ids']) if (len(sentence) > cfg.min_src_ntokens)]

        # Keep only relevant sentences. Note they are already trimmed by the tokenizer
        src = [src['input_ids'][i] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:cfg.max_nsents]
        labels = labels[:cfg.max_nsents]

        # Flatten into a single sequence (sents will be separated by [SEP] and [CLS] tokens already)
        src = list(itertools.chain(*src))
        if len(src) > 512:
            src = src[:511] + [tokenizer.sep_token_id] # Truncate to 512 tokens

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
        inputs['segment_ids'].append(segment_ids)
        inputs['cls_ids'].append(cls_ids)
        inputs['labels'].append(labels)

        # Append the preprocessed sample to the batch_inputs dictionary
        for key in inputs_keys:
            batch_inputs[key].extend(inputs[key])

    batch_inputs['input_ids'], batch_inputs['mask'] = pad_sequence(batch_inputs['input_ids'], padding_value=tokenizer.pad_token_id, max_length=512)
    batch_inputs['labels'], _ = pad_sequence(batch_inputs['labels'], padding_value=-1, max_length=cfg.max_nsents)
    batch_inputs['segment_ids'], _ = pad_sequence(batch_inputs['segment_ids'], padding_value=0, max_length=512)
    batch_inputs['cls_ids'], batch_inputs['mask_cls'] = pad_sequence(batch_inputs['cls_ids'], padding_value=0, max_length=cfg.max_nsents)
    return batch_inputs

def preprocess_validation(batch, tokenizer, cfg):
    """
    Simple preprocessing of the data for the extractive summarization task evaluation.
    Assumes that the data is a Dataset with the keys: 'src', 'tgt', and 'labels'.
    `cfg` is a configuration object with the following attributes:
        - max_src_ntokens: maximum number of tokens in the source text
        - min_src_ntokens: minimum number of tokens in the source text
        - max_nsents: maximum number of sentences in the source text
        - min_nsents: minimum number of sentences in the source text
    Returns the preprocessed batch ready for the model.
    """
    # Get inputs keys
    inputs_keys = ['input_ids', 'labels', 'segment_ids', 'cls_ids', 'mask', 'mask_cls', 'src_ids', 'tgt_ids']
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

        # Track the original text and target summary
        orig_src = tokenizer(
            src,
            max_length=None,
            stride=0,
            return_token_type_ids=False,
            return_attention_mask=False
        )
        inputs['src_ids'].append(list(itertools.chain(*orig_src['input_ids']))) # flattened

        tgt = tokenizer(
            ' '.join(tgt),
            max_length=None, ## 2000, truncation=True
            stride=0,
            return_token_type_ids=False,
            return_attention_mask=False
        )
        inputs['tgt_ids'].append(tgt['input_ids']) # flattened

        del orig_src, tgt

        # Prepare the right input for BERT
        src = tokenizer(
            src,
            max_length=cfg.max_src_ntokens,
            truncation=True,
            stride=0,
            return_token_type_ids=False,
            return_attention_mask=False
        )

        # Ignore senteces that are too short
        # *Assumption*: if sentence is short it is not relevant
        idxs = [i for i, sentence in enumerate(src['input_ids']) if (len(sentence) > cfg.min_src_ntokens)]

        # Trim sentences to a maximum. Note they are already trimmed by the tokenizer
        src = [src['input_ids'][i] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:cfg.max_nsents]
        labels = labels[:cfg.max_nsents]

        # Flatten into a single sequence (sents will be separated by [SEP] and [CLS] tokens already)
        src = list(itertools.chain(*src))
        if len(src) > 512:
            src = src[:511] + [tokenizer.sep_token_id] # Truncate to 512 tokens

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

        # Append the preprocessed sample to the batch_inputs dictionary
        for key in inputs_keys:
            batch_inputs[key].extend(inputs[key])

    batch_inputs['input_ids'], batch_inputs['mask'] = pad_sequence(batch_inputs['input_ids'], padding_value=tokenizer.pad_token_id, max_length=None) # 512
    batch_inputs['labels'], _ = pad_sequence(batch_inputs['labels'], padding_value=-1, max_length=None) # cfg.max_nsents
    batch_inputs['segment_ids'], _ = pad_sequence(batch_inputs['segment_ids'], padding_value=0, max_length=None) # 512
    batch_inputs['cls_ids'], batch_inputs['mask_cls'] = pad_sequence(batch_inputs['cls_ids'], padding_value=0, max_length=None) # cfg.max_nsents
    batch_inputs['src_ids'], _ = pad_sequence(batch_inputs['src_ids'], padding_value=tokenizer.pad_token_id, max_length=None)
    batch_inputs['tgt_ids'], _ = pad_sequence(batch_inputs['tgt_ids'], padding_value=tokenizer.pad_token_id, max_length=None) # 2000
    return batch_inputs

def pad_sequence(sequences, padding_value=0, max_length=None):
    """
    Auxiliary function to pad a list of sequences to the same length.
    """
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    padded_sequences = [seq + [padding_value] * (max_length - len(seq)) for seq in sequences]
    mask = [[1] * len(seq) + [0] * (max_length - len(seq)) for seq in sequences]
    return padded_sequences, mask

# Data loader #
class SummarizerDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(value) for key, value in self.data[idx].items()}
        return item
    
def collate_batch(batch, pad_token_id):
    """
    Special collator to ensure that the batch is padded to the maximum length of the batch according to the different keys.
    """
    keys = batch[0].keys()
    padded_batch = {}

    for k in keys:
        if k in ['labels']:
            padded_batch[k] = torch_pad_sequence([x[k] for x in batch], batch_first=True, padding_value=-1)
        elif k in ['segment_ids', 'cls_ids', 'mask', 'mask_cls']:
            padded_batch[k] = torch_pad_sequence([x[k] for x in batch], batch_first=True, padding_value=0)
        else:
            padded_batch[k] = torch_pad_sequence([x[k] for x in batch], batch_first=True, padding_value=pad_token_id)

    del batch
    return padded_batch