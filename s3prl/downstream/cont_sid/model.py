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


        """
        encoder module
        """
        self.encoder = self.get_encoder(input_dim, kwargs['encoder_module'], kwargs['encoder_config'])
        latest_dim = kwargs['encoder_config']['agg_dim']
        #self.pre_net = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation, weight_number=kwargs['weight_number'])
        self.post_net = get_downstream_model(latest_dim, output_dim, post_net)        

    def get_encoder(self, input_dim, module, config):
        #創建模型, load parameter, 把grad關掉

        encoder = eval(module)(input_dim=input_dim, agg_dim=config['agg_dim'], dropout_p=config['dropout_p'], batch_norm=config['batch_norm'])
        
        #load parameter
        encoder_state_dict = encoder.state_dict()
        previous_ckpt_dict = torch.load(config['ckpt_path'])['Downstream']
        filtered_dict = {}
        for k in encoder_state_dict.keys():
            for prev_k in previous_ckpt_dict.keys():
                if k in prev_k:
                    filtered_dict[k] = previous_ckpt_dict[prev_k]
                    break
        
        encoder_state_dict.update(filtered_dict)
        encoder.load_state_dict(encoder_state_dict)

        #turn out grad
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder
        


    def forward(self, hidden_state, features_len=None):
        # if self.pre_net is not None:
        #     hidden_state, features_len = self.pre_net(hidden_state, features_len)
        
        hidden_state = self.encoder(hidden_state)
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

    def forward(self, feature_BxTxH, **kwargs):

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

