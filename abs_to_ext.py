#!/usr/bin/env python
"""
Script to convert abstractive summaries to extractive summaries using multiprocessing.
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

import re
import time
import json
import argparse
import itertools
from tqdm import tqdm
from multiprocessing import Pool
from utils import _get_word_ngrams, cal_rouge, cal_rouge_l, tokenize_text_to_sentences, tokenize_sentences_to_words

from datasets import Dataset, load_dataset, load_from_disk

# Basic method to select sentences for extractive summarization #
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    """
    Select the sentences that maximize ROUGE-1 + ROUGE-2 scores.
    It checks the combination of sentences in the given order, so is biased to the first sentences.
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            # rouge_l = cal_rouge_l(abstract, sents[i])['f']
            rouge_score = rouge_1 + rouge_2# + rouge_l
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    """
    Select the sentences that maximize ROUGE-1 + ROUGE-2 scores.
    It checks the combination of all sentences.
    Do not even try to run this function with more than `n` sentences.
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))

# Get positive labels #
def abstractive_to_extractive(data, src_name, tgt_name, summary_size, method, comb_max_sents=10):
    """
    Get the sentences that maximize the ROUGE-1 + ROUGE-2 scores.
    `method` can be 'greedy', 'combination', or an in between (`comb_max_sents` is the threshold).
    """
    # Split source and target into sentences
    orig_src = tokenize_text_to_sentences(data[src_name])
    src = tokenize_sentences_to_words(orig_src)
    orig_tgt = tokenize_text_to_sentences(data[tgt_name])
    tgt = tokenize_sentences_to_words(orig_tgt)

    # Get positive labels
    labels = [0] * len(src) # One label for each sentence
    if method == 'greedy':
        oracle_ids = greedy_selection(src, tgt, summary_size)
    elif method =='combination':
        oracle_ids = combination_selection(src, tgt, summary_size)
    else:
        if len(src) > comb_max_sents:
            oracle_ids = greedy_selection(src, tgt, summary_size)
        else:
            start_time = time.time()
            oracle_ids = combination_selection(src, tgt, summary_size)
    
    for idx in oracle_ids:
        labels[idx] = 1

    del src, tgt
    return {
        'src': orig_src,
        'tgt': orig_tgt,
        'labels': labels
    }

def main(data, src_name, tgt_name, summary_size, method):
    # Define the number of workers to use in parallel processing
    num_workers = os.cpu_count()

    # Create a generator expression for the arguments
    args_generator = ((sample, src_name, tgt_name, summary_size, method) for sample in data)

    # Use a multiprocessing.Pool to run the preprocess function in parallel
    with Pool(processes=num_workers) as pool:
        # Use tqdm to track the progress of the parallel processing
        results = list(pool.starmap(abstractive_to_extractive, tqdm(args_generator, total=len(data))))

    # 'results' will be a list containing the output of your preprocess function for each sample in the dataset
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data using abstractive_to_extractive function.')
    parser.add_argument('-f', '--data_path', type=str, default='cnn_dailymail', help='Path to the dataset.')
    parser.add_argument('-s', '--src_name', type=str, default='article', help='Name of the source field in the input dataset.')
    parser.add_argument('-t', '--tgt_name', type=str, default='highlights', help='Name of the target field in the input dataset.')
    parser.add_argument('-sz', '--summary_size', type=int, default=3, help='Maximum summary size.')
    parser.add_argument('-m', '--method', type=str, default='greedy', help='Method to use for selecting sentences. (greedy or combination)')
    parser.add_argument('-c', '--data_config', type=str, default=None, help='Config of the Hugging Face dataset.')
    parser.add_argument('-hf', '--from_hub', action='store_true', default=True, help='Load dataset from Hugging Face.')
    parser.add_argument('-o', '--output_dir', type=str, default='processed_data', help='Output directory for the processed dataset.')

    args = parser.parse_args()

    if args.from_hub:
        # Load data from Hugging Face
        dataset = load_dataset(args.data_path, name=args.data_config)
        for split in dataset.keys():
            # Preprocess the data for the current split
            print(f'Processing {split} split...')
            results = main(dataset[split], args.src_name, args.tgt_name, args.summary_size, args.method)

            # Save the results to a new Hugging Face dataset
            print(f'Saving {split} split...')
            processed_dataset = Dataset.from_dict({
                'src': [item['src'] for item in results],
                'tgt': [item['tgt'] for item in results],
                'labels': [item['labels'] for item in results],
            })
            processed_dataset.save_to_disk(os.path.join(args.output_dir, split))

        # Create dataset_dict.json file
        with open(os.path.join(args.output_dir, 'dataset_dict.json'), 'w') as f:
            json.dump({'splits': [split for split in dataset.keys()]}, f)

        print('Done!')
    else:
        print('Not implemented yet.')