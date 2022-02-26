import torch
import torch.nn as nn
import torch.nn.functional as F


def get_downstream_model(input_dim, output_dim, config):
    model_cls = eval(config['select'])
    model_conf = config.get(config['select'], {})
    model = model_cls(input_dim, output_dim, **model_conf)
    return model


class FrameLevel(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens=None, activation='ReLU', **kwargs):
        super().__init__()
        latest_dim = input_dim
        self.hiddens = []
        if hiddens is not None:
            for dim in hiddens:
                self.hiddens += [
                    nn.Linear(latest_dim, dim),
                    getattr(nn, activation)(),
                ]
                latest_dim = dim
        self.hiddens = nn.Sequential(*self.hiddens)
        self.linear = nn.Linear(latest_dim, output_dim)

    def forward(self, hidden_state, features_len=None):
        hidden_states = self.hiddens(hidden_state)
        logit = self.linear(hidden_state)

        return logit, features_len


class UtteranceLevel(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        pooling='MeanPooling',
        activation='ReLU',
        pre_net=None,
        post_net={'select': 'FrameLevel'},
        **kwargs
    ):
        super().__init__()
        
        latest_dim = input_dim
        self.pre_net = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation, weight_number=kwargs['weight_number'])
        self.post_net = get_downstream_model(latest_dim, output_dim, post_net)
        


    def forward(self, hidden_state, features_len=None):
        if self.pre_net is not None:
            hidden_state, features_len = self.pre_net(hidden_state, features_len)

        pooled, features_len = self.pooling(hidden_state, features_len)
        logit, features_len = self.post_net(pooled, features_len)

        return logit, features_len
    
class FramewisePooling(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        #add by chiluen
        
        self.weights = nn.Parameter(torch.ones(kwargs['weight_number']))

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        ##這邊要先切chunk, 然後乘上weight, 最後接起來
        if len(self.weights) != 10:
            agg_vec_list = []
            for i in range(len(feature_BxTxH)): #取batch
                temp_agg_vec_list = []
                temp_feature = torch.chunk(feature_BxTxH[i][:features_len[i]], len(self.weights))
                norm_weights = F.softmax(self.weights, dim=-1)
                for j in range(len(self.weights)): #算weight
                    temp_agg_vec_list.append(temp_feature[j] * norm_weights[j])
                agg_vec_list.append(torch.cat(temp_agg_vec_list).sum(dim=0))
            return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()
      
        else: #沒有需要用到weight
        
            agg_vec_list = []
            for i in range(len(feature_BxTxH)):
                agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
                agg_vec_list.append(agg_vec)

            return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()
    


class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()


class AttentivePooling(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, input_dim, activation, **kwargs):
        super(AttentivePooling, self).__init__()
        self.sap_layer = AttentivePoolingModule(input_dim, activation)

    def forward(self, feature_BxTxH, features_len):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)

        return sap_vec, torch.ones(len(feature_BxTxH)).long()


class AttentivePoolingModule(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
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
