#!/usr/bin/env python
'''
Simple BERT-based summarizer for extractive summarization.
'''
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel

class Classifier(nn.Module):
    """
    Simple classifier to predict the probability of each sentence to be included in the summary.
    """
    def __init__(self, hidden_size, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores
    
class BertSummarizerConfig(BertConfig):
    """
    Configuration class to store the configuration of a `BertSummarizer`.
    Inherits from `BertConfig` and loads the BERT checkpoint.
    """
    def __init__(self, checkpoint=None, **kwargs):
        super(BertSummarizerConfig, self).__init__(**kwargs)
        self.checkpoint = checkpoint

class BertSummarizer(BertPreTrainedModel):
    """
    Architecture to fine tune BERT for extractive summarization.
    BERT is used to encode the sentences.
    Afterward, a simple linear layer is used to predict the probability of each sentence to be included in the summary.
    """
    config_class = BertSummarizerConfig
    base_model_prefix = 'bert'
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(config.checkpoint) # Load pretrained bert
        self.encoder = Classifier(self.bert.config.hidden_size) # Add a linear layer on top of BERT for classification

        # Initialize encoder weights
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -1.0, 1.0)
                
    def forward(self, input_ids=None, segment_ids=None, cls_ids=None, mask=None, mask_cls=None, labels=None, src_ids=None, tgt_ids=None):
        """
        The last hidden state of the BERT is used to encode the sentences.
        The first token of each sentence is used as a representation of the sentence.
        The representation of each sentence is then used to predict the probability of each sentence to be included in the summary.
        """
        top_vec = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=mask).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_ids]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return  {'logits': sent_scores,
                 'mask_cls': mask_cls}