# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace

#########
# MODEL #
#########

def decide_utter_input_dim(agg_module_name, input_dim, agg_dim):		
    if agg_module_name =="ASP":		
        utter_input_dim = input_dim*2		
    elif agg_module_name == "SP":		
        # after aggregate to utterance vector, the vector hidden dimension will become 2 * aggregate dimension.		
        utter_input_dim = agg_dim*2		
    elif agg_module_name == "MP":		
        utter_input_dim = agg_dim		
    else:		
        utter_input_dim = input_dim		
    return utter_input_dim

# Pooling Methods

class MP(nn.Module):

    def __init__(self, **kwargs):
        super(MP, self).__init__()
        # simply MeanPooling / no additional parameters

    def forward(self, feature_BxTxH, att_mask_BxT, **kwargs):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            if torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature_BxTxH[i])
            else:
                length = torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False)[0] + 1
            agg_vec=torch.mean(feature_BxTxH[i][:length], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)

class AP(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, out_dim, input_dim):
        super(AP, self).__init__()

        # Setup
        self.linear = nn.Linear(input_dim, out_dim)
        self.sap_layer = AttentivePooling(out_dim)
        self.act_fn=nn.ReLU()
    
    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        #Encode
        feature_BxTxH = self.linear(feature_BxTxH)
        sap_vec, _ = self.sap_layer(feature_BxTxH, att_mask_BxT)

        return sap_vec

class ASP(nn.Module):
    ''' Attentive Statistic Pooling module incoporate attention mask'''

    def __init__(self, out_dim, input_dim):
        super(ASP, self).__init__()

        # Setup
        self.linear = nn.Linear(input_dim, out_dim)
        self.ap_layer = AttentivePooling(out_dim)

    
    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            att_mask_BxT  - [BxT]     Attention Mask logits
        '''
        #Encode
        feature_BxTxH = self.linear(feature_BxTxH)
        sap_vec, att_w = self.ap_layer(feature_BxTxH, att_mask_BxT)
        variance = torch.sqrt(torch.sum(att_w * feature_BxTxH * feature_BxTxH, dim=1) - sap_vec**2 + 1e-8)
        statistic_pooling = torch.cat([sap_vec, variance], dim=-1)

        return statistic_pooling

class SP(nn.Module):
    ''' Statistic Pooling incoporate attention mask'''

    def __init__(self, out_dim, input_dim, *kwargs):
        super(SP, self).__init__()

        # Setup
        self.mp_layer = MP()
    
    def forward(self, feature_BxTxH, att_mask_BxT):

        ''' 
        Arguments
            feature - [BxTxH]   Acoustic feature with shape 
            att_mask- [BxT]     Attention Mask logits
        '''
        #Encode
        mean_vec = self.mp_layer(feature_BxTxH, att_mask_BxT)
        variance_vec_list = []
        for i in range(len(feature_BxTxH)):
            if torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature_BxTxH[i])
            else:
                length = torch.nonzero(att_mask_BxT[i] < 0, as_tuple=False)[0] + 1
            variances = torch.std(feature_BxTxH[i][:length], dim=-2)
            variance_vec_list.append(variances)
        var_vec = torch.stack(variance_vec_list)

        statistic_pooling = torch.cat([mean_vec, var_vec], dim=-1)

        return statistic_pooling

class AttentivePooling(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, **kwargs):
        super(AttentivePooling, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax
    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w


# General Interface
class Model(nn.Module):
    def __init__(self, input_dim, agg_dim, agg_module_name, module_name, utterance_module_name, hparams):
        super(Model, self).__init__()
        
        # support for XVector(standard architecture)
        # Framewise FeatureExtractor
        extractor_config = {**hparams, **{"input_dim": input_dim}}
        self.framelevel_feature_extractor= eval(module_name)(**extractor_config)

        # agg_module: 
        # current support:
        # [ "AP" (Attentive Pooling), "MP" (Mean Pooling), "SP" (Statistic Pooling), "SAP" (Statistic Attentive Pooling) ]
        agg_module_config = {"out_dim": input_dim, "input_dim": agg_dim}
        self.agg_method = eval(agg_module_name)(**agg_module_config)

        utterance_input_dim = decide_utter_input_dim(agg_module_name=agg_module_name, agg_dim=agg_dim, input_dim=input_dim)

        # after extract utterance level vector, put it to utterance extractor (XVector Architecture)
        utterance_extractor_config = {"input_dim": utterance_input_dim,"out_dim": input_dim}
        self.utterancelevel_feature_extractor= eval(utterance_module_name)(**utterance_extractor_config)

    def forward(self, features_BxTxH, att_mask_BxT):

        total_num = features_BxTxH.shape[0] // 2
        original_features, original_att_mask = features_BxTxH[:total_num], att_mask_BxT[:total_num]
        augment_features, augment_att_mask = features_BxTxH[total_num:], att_mask_BxT[total_num:]

        #for original
        original_features = self.framelevel_feature_extractor(original_features, original_att_mask[:,None,None])
        original_utterance_vector = self.agg_method(original_features, original_att_mask)
        original_utterance_vector = self.utterancelevel_feature_extractor(original_utterance_vector)

        #for augment
        augment_features = self.framelevel_feature_extractor(augment_features, augment_att_mask[:,None,None])
        augment_utterance_vector = self.agg_method(augment_features, augment_att_mask)
        augment_utterance_vector = self.utterancelevel_feature_extractor(augment_utterance_vector)

        return original_features, original_utterance_vector, augment_features, augment_utterance_vector

class UtteranceExtractor(nn.Module):
    def __init__(self, input_dim, out_dim, **kwargs):
        super(UtteranceExtractor,self).__init__()
        self.linear1 = nn.Linear(input_dim,out_dim)
        self.linear2 = nn.Linear(out_dim,out_dim)
        self.act_fn = nn.ReLU()
    def forward(self, x_BxH):
        hid_BxH = self.linear1(x_BxH)
        hid_BxH = self.act_fn(hid_BxH)
        hid_BxH = self.linear2(hid_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH
    
    def inference(self, feature_BxH):
        hid_BxH = self.linear1(feature_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH

class XVector(nn.Module):
    def __init__(self, input_dim, agg_dim, dropout_p, batch_norm, **kwargs):
        super(XVector, self).__init__()
        # simply take mean operator / no additional parameters
        self.module = nn.Sequential(
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=5, dilation=1, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=3, dilation=2, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=3, dilation=3, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=input_dim, context_size=1, dilation=1, batch_norm=batch_norm, dropout_p=dropout_p),
            TDNN(input_dim=input_dim, output_dim=agg_dim, context_size=1, dilation=1, batch_norm=batch_norm, dropout_p=dropout_p),
        )

    def forward(self, feature_BxTxH, att_mask_BxTx1x1, **kwargs):

        feature_BxTxH=self.module(feature_BxTxH)
        return feature_BxTxH

class TDNN(nn.Module):
        
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=False,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x_BxTxH):
        '''
        input: size (batch B, seq_len T, input_features H)
        outpu: size (batch B, new_seq_len T*, output_features H)
        '''

        _, _, d = x_BxTxH.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x_BxTxH = x_BxTxH.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x_BxTxH = F.unfold(
                        x_BxTxH, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x_BxTxH = x_BxTxH.transpose(1,2)
        x_BxTxH = self.kernel(x_BxTxH)
        x_BxTxH = self.nonlinearity(x_BxTxH)
        
        if self.dropout_p:
            x_BxTxH = self.drop(x_BxTxH)

        if self.batch_norm:
            x_BxTxH = x_BxTxH.transpose(1,2)
            x_BxTxH = self.bn(x_BxTxH)
            x_BxTxH = x_BxTxH.transpose(1,2)

        return x_BxTxH
