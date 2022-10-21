import torch
import torch.nn as nn
from layers.attention import Attention
from layers.point_wise_feed_forward import PositionwiseFeedForward

class CONTEXT_BERT(nn.Module):
    def __init__(self,bert):
        super(CONTEXT_BERT, self).__init__()

        dropout=0.1
        bert_dim=768 #raberta # bert_dim=1024#roberta-large
        hidden_dim=300
        polarities_dim=3        
        self.bert=bert
        self.dropout = nn.Dropout(dropout)
        self.attn_k = Attention(bert_dim, out_dim=hidden_dim, n_head=8, score_function='mlp', dropout=dropout)
        self.attn_q = Attention(bert_dim, out_dim=hidden_dim, n_head=8, score_function='mlp', dropout=dropout)
        self.ffn_c = PositionwiseFeedForward(hidden_dim, dropout=dropout)
        self.ffn_t = PositionwiseFeedForward(hidden_dim, dropout=dropout)

        self.attn_s1 = Attention(hidden_dim, n_head=8, score_function='mlp', dropout=dropout)
        self.dense = nn.Linear(hidden_dim*3, polarities_dim)
        self.softmax=nn.Softmax(dim=1)
 
        
    def forward(self, inputs):
        context_len=128
        target_len=128
        
        a_input_ids,a_input_mask,b_input_ids,b_input_mask=inputs

        target,_=self.bert(a_input_ids, token_type_ids=None, attention_mask=a_input_mask,)
        context,_=self.bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,)
        context = self.dropout(context)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(target, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len)
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len)
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len)
        
        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out=self.dense(x)
        return out
