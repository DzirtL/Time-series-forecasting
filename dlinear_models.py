import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import IPython
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        #print("logits,mask:",attn_logits.shape,mask.shape)
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        #IPython.embed()
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention



# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(0)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = scaled_dot_product(query, key, value, mask=mask)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads,
                 dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim,
                        input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn)
        x = self.norm1(x)

        linear = self.linear_net(x)
        x = x + self.dropout(linear)
        x = self.norm2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads,
                 dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadedAttention(num_heads,input_dim,
                                             dropout)
        self.src_attn = MultiHeadedAttention(num_heads,input_dim,
                                             dropout)

        self.linear_net_src = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.linear_net_tgt = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.norm1_src = nn.LayerNorm(input_dim)
        self.norm2_src = nn.LayerNorm(input_dim)
        self.norm1_tgt = nn.LayerNorm(input_dim)
        self.norm2_tgt = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory
        attn = self.self_attn(x,x,x,mask=tgt_mask)
        x = x+self.dropout(attn)
        x = self.norm1_src(x)
        linear = self.linear_net_src(x)
        x = x+self.dropout(linear)
        x = self.norm2_src(x)


        attn = self.src_attn(x,m,m,mask=src_mask)
        x = x+self.dropout(attn)
        x = self.norm1_tgt(x)
        linear = self.linear_net_tgt(x)
        x = x+self.dropout(linear)
        x = self.norm2_tgt(x)
        
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x,memory,src_mask,tgt_mask):
        for l in self.layers:
            x = l(x,memory,src_mask,tgt_mask)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


#temporal embedding 
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        #Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        Embed = nn.Embedding
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        #print("temporal embedding got data:",x.shape,"values:",torch.max(x[0],dim=1)[0])
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        print("x in token embedding:",x.shape)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask



class Transformer_base_detailed(nn.Module):
    """ Based on custom modules """
    def __init__(self,cfg):
        super().__init__()

        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        self.encoder = TransformerEncoder(num_layers=cfg["enc_numlayers"],input_dim=cfg["embed_dim"],num_heads=cfg["num_heads"],dim_feedforward=cfg["embed_dim"])
        self.decoder = TransformerDecoder(num_layers=cfg["dec_numlayers"],input_dim=cfg["embed_dim"],num_heads=cfg["num_heads"],dim_feedforward=cfg["embed_dim"])

    def forward(self,src,tgt):
        """
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]
        outputs:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # [bs, src_seq_len, embed_dim]
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)

        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
           

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        #tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        tgt_mask = subsequent_mask(tgt.shape[1]).to(tgt.device)

        # [bs, tgt_seq_len, embed_dim]
        pred_encoder = self.encoder(src)
        pred_decoder = self.decoder(tgt,pred_encoder,src_mask=None,tgt_mask=tgt_mask) 
        pred = self.output_layer(pred_decoder)

        return pred




class Transformer_base(nn.Module):
    """ Based on PyTorch nn.Transformer module """
    def __init__(self,cfg):
        super().__init__()

        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        self.transformer = nn.Transformer(nhead=cfg["num_heads"],num_encoder_layers=cfg["enc_numlayers"],
                                          num_decoder_layers=cfg["dec_numlayers"],
                                          d_model=cfg["embed_dim"],dropout=cfg["dropout"],batch_first=True)


    def forward(self,src,tgt):
        """
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]
        outputs:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # [bs, src_seq_len, embed_dim]
        #print("input:",src.shape,"tgt:",tgt.shape)

        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)

        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
           

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        # [bs, tgt_seq_len, embed_dim]

        # [bs, tgt_seq_len, embed_dim]
        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        return pred

class Transformer_MT_S1(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg=cfg
        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"]*cfg["input_slice"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        self.transformer = nn.Transformer(nhead=cfg["num_heads"],num_encoder_layers=cfg["enc_numlayers"],
                                          num_decoder_layers=cfg["dec_numlayers"],
                                          d_model=cfg["embed_dim"],dropout=cfg["dropout"],batch_first=True)


    def forward(self,src_,tgt):
        src2 = []
        for i in range(self.cfg["input_slice"],src_.shape[1]):
            src2+=[src_[:,i-self.cfg["input_slice"]:i].reshape(src_.shape[0],1,-1)]
        src2 = torch.cat(src2,dim=1)
        
        src = self.encoder_embedding(src2)
        src = self.positional_encoding(src)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        return pred


class Transformer_MT_VariableStep(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg["input_dim"]
        self.embed_dim = cfg["embed_dim"]
        self.patch_len = cfg.get("input_slice", 24) 
        self.stride = cfg.get("stride", 1)          

        self.patch_embedding = nn.Conv1d(
            in_channels=self.input_dim, 
            out_channels=self.embed_dim, 
            kernel_size=self.patch_len, 
            stride=self.stride
        )
        
        self.dec_embedding = nn.Linear(cfg["out_dim"], self.embed_dim)
        
        self.src_pe = nn.Parameter(torch.zeros(1, 5000, self.embed_dim))
        self.tgt_pe = nn.Parameter(torch.zeros(1, 5000, self.embed_dim))

        self.transformer = nn.Transformer(
            d_model=self.embed_dim, 
            nhead=cfg["num_heads"], 
            num_encoder_layers=cfg["enc_numlayers"], 
            num_decoder_layers=cfg["dec_numlayers"], 
            dim_feedforward=self.embed_dim * 4, 
            dropout=cfg["dropout"], 
            batch_first=True
        )
        
        self.predictor = nn.Linear(self.embed_dim, cfg["out_dim"])
        
    def forward(self, src, tgt):
        src_conv = src.permute(0, 2, 1) 
        src_emb = self.patch_embedding(src_conv)
        src_emb = src_emb.permute(0, 2, 1)
        
        tgt_emb = self.dec_embedding(tgt)
        
        src_emb = src_emb + self.src_pe[:, :src_emb.size(1), :]
        tgt_emb = tgt_emb + self.tgt_pe[:, :tgt_emb.size(1), :]
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        out = self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask)
        return self.predictor(out)
    

class Transformer_MT_S1_decode(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg=cfg
        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"]*cfg["input_slice"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg["embed_dim"],nhead=cfg["num_heads"],batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer,num_layers=cfg["dec_numlayers"])


    def forward(self,src_,tgt):
        """
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]
        outputs:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # [bs, src_seq_len, embed_dim]
        #print("input:",src.shape,"tgt:",tgt.shape)

        src2 = []
        for i in range(self.cfg["input_slice"],src_.shape[1]):
            src2+=[src_[:,i-self.cfg["input_slice"]:i].reshape(src_.shape[0],1,-1)]
        src2 = torch.cat(src2,dim=1)
        
        src = self.encoder_embedding(src2)
        src = self.positional_encoding(src)

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        # [bs, tgt_seq_len, embed_dim]
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        # [bs, tgt_seq_len, embed_dim]
        pred = self.transformer(tgt,src, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        return pred


class Transformer_MT_S1_decode_detailed(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg=cfg
        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"]*cfg["input_slice"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        #decoder_layer = nn.TransformerDecoderLayer(d_model=cfg["embed_dim"],nhead=cfg["num_heads"],batch_first=True)
        #self.transformer = nn.TransformerDecoder(decoder_layer,num_layers=cfg["dec_numlayers"])
        self.decoder = TransformerDecoder(num_layers=cfg["dec_numlayers"],input_dim=cfg["embed_dim"],num_heads=cfg["num_heads"],dim_feedforward=cfg["embed_dim"])



    def forward(self,src_,tgt):
        """
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]
        outputs:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # [bs, src_seq_len, embed_dim]
        #print("input:",src.shape,"tgt:",tgt.shape)

        src2 = []
        for i in range(self.cfg["input_slice"],src_.shape[1]):
            src2+=[src_[:,i-self.cfg["input_slice"]:i].reshape(src_.shape[0],1,-1)]
        src2 = torch.cat(src2,dim=1)
        
        src = self.encoder_embedding(src2)
        src = self.positional_encoding(src)

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        #tgt_mask_ = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        tgt_mask = subsequent_mask(tgt.shape[1]).to(tgt.device)


        
        # [bs, tgt_seq_len, embed_dim]
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        # [bs, tgt_seq_len, embed_dim]
        pred = self.decoder(tgt,src,src_mask=None,tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        return pred



class Transformer_MT_S24(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg=cfg
        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"]*cfg["input_slice"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        self.transformer = nn.Transformer(nhead=cfg["num_heads"],num_encoder_layers=cfg["enc_numlayers"],
                                          num_decoder_layers=cfg["dec_numlayers"],
                                          d_model=cfg["embed_dim"],dropout=cfg["dropout"],batch_first=True)


    def forward(self,src,tgt):
        """
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]
        outputs:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # [bs, src_seq_len, embed_dim]
        #print("input:",src.shape,"tgt:",tgt.shape)

        src2 = []
        for i in range(0,src.shape[1]-self.cfg["input_slice"],self.cfg["input_slice"]):
            src2+=[src[:,i:i+self.cfg["input_slice"]].reshape(src.shape[0],1,-1)]
        src2 = torch.cat(src2,dim=1)
        #print("src2:",src2.shape)

        
        src = self.encoder_embedding(src2)
        src = self.positional_encoding(src)

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        tgt_mask = generate_subsequent_mask(tgt.shape[1],tgt.shape[1],src.dtype,src.device)
        # [bs, tgt_seq_len, embed_dim]
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        #print("src,tar,mask:",src.shape,tgt.shape,tgt_mask.shape)
        # [bs, tgt_seq_len, embed_dim]
        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        return pred




class Transformer_trend(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg = cfg
        self.positional_encoding = PositionalEncoding(cfg["embed_dim"])
        self.temporal_encoding = TemporalEmbedding(cfg["embed_dim"]) if cfg["temporal_encoding"]==True else None

        self.encoder_embedding = torch.nn.Linear(cfg["input_dim"], cfg["embed_dim"])
        self.decoder_embedding = torch.nn.Linear(cfg["out_dim"], cfg["embed_dim"])

        self.output_layer = torch.nn.Linear(cfg["embed_dim"], cfg["out_dim"])

        self.transformer = nn.Transformer(nhead=cfg["num_heads"],num_encoder_layers=cfg["enc_numlayers"],
                                          num_decoder_layers=cfg["dec_numlayers"],
                                          d_model=cfg["embed_dim"],batch_first=True)


        self.decomp = series_decomp(25)
        self.trend = nn.Linear(cfg["input_dim"]*cfg["inp_seq_len"],cfg["out_dim"]*cfg["outp_seq_len"])

    def forward(self,src_,tgt_):
        """
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]
        outputs:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # [bs, src_seq_len, embed_dim]
        #print("input:",src_.shape,"tgt:",tgt_.shape)

        src,trend_src = self.decomp(src_)
        tgt,trend_tgt = self.decomp(tgt_)

        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)

        tgt = self.decoder_embedding(tgt_)
        tgt = self.positional_encoding(tgt)

        trend_out = self.trend(trend_src.reshape(trend_src.shape[0],-1))
        trend_out = trend_out.reshape(src.shape[0],-1,self.cfg["out_dim"])


        #print("src,tgt:",src.shape,tgt.shape,trend_src.shape,trend_tgt.shape)

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        # [bs, tgt_seq_len, embed_dim]

        # [bs, tgt_seq_len, embed_dim]
        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        self.trend_out = trend_out

        return pred+trend_out[:,:pred.shape[1]]






class DLinear(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg["num_layers"]==1:
            self.linear = nn.Linear(cfg["input_dim"]*cfg["inp_seq_len"],
                                    cfg["out_dim"]*cfg["outp_seq_len"])
        else:
            modules = [nn.Linear(cfg["input_dim"]*cfg["inp_seq_len"],cfg["embed_dim"]*cfg["inp_seq_len"]),
                       nn.ReLU(),nn.Linear(cfg["embed_dim"]*cfg["inp_seq_len"],
                                           cfg["out_dim"]*cfg["outp_seq_len"])]
            self.linear = nn.Sequential(*modules)
    def forward(self,src,tgt):
        out = self.linear(src.reshape(src.shape[0],-1))
        return out.reshape(src.shape[0],-1,1)

class DLinear_MHE(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.linear = nn.Linear(cfg["input_dim"]*cfg["inp_seq_len"],cfg["out_dim"]*1)
    def forward(self,src,tgt):
        inp = torch.cat([src,tgt],dim=1)
        out = []
        for i in range(tgt.shape[1]):
            tmp = inp[:,i:i+src.shape[1]]
            out+=[self.linear(tmp.reshape(src.shape[0],-1))]
            out[-1] = out[-1].reshape(out[-1].shape[0],-1,1)
        out = torch.cat(out,dim=1) 
        return out 



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear_trend(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg = cfg
        self.decomp = series_decomp(25)

        self.linear = nn.Linear(cfg["input_dim"]*cfg["inp_seq_len"],cfg["out_dim"]*cfg["outp_seq_len"])
        self.trend= nn.Linear(cfg["input_dim"]*cfg["inp_seq_len"],cfg["out_dim"]*cfg["outp_seq_len"])


    def forward(self,src,tgt):
        linear,trend = self.decomp(src)
        linear_out = self.linear(linear.reshape(linear.shape[0],-1))
        trend_out = self.trend(trend.reshape(trend.shape[0],-1))

        return (linear_out+trend_out).reshape(src.shape[0],self.cfg["outp_seq_len"],self.cfg["out_dim"])

class Transformer_EncoderFree(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg["embed_dim"]

        self.embedding = nn.Linear(cfg["input_dim"], self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=cfg["num_heads"], 
            dim_feedforward=self.embed_dim * 4, 
            dropout=cfg["dropout"], 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.get("dec_numlayers", 2))

        self.output_layer = nn.Linear(self.embed_dim, cfg["out_dim"])

    def forward(self, src, tgt):
        seq = torch.cat([src, tgt], dim=1)
        
        seq_emb = self.embedding(seq)
        seq_emb = self.positional_encoding(seq_emb)
        
        mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)
        
        out = self.transformer(seq_emb, mask=mask, is_causal=True)
        pred = self.output_layer(out)
        
        return pred[:, src.size(1):, :]
    
    
#tests of modules
if __name__ == "__main__":
    import sys
    key = sys.argv[1]

    if key == "scaled_dot" or key == "all":
        #test of scaled dot product
        seq_len, d_k = 3, 2
        q = torch.randn(seq_len, d_k)
        k = torch.randn(seq_len, d_k)
        v = torch.randn(seq_len, d_k)
        values, attention = scaled_dot_product(q, k, v)
        print("Q\n", q)
        print("K\n", k)
        print("V\n", v)
        print("Values\n", values)
        print("Attention\n", attention)

    if key == "encoding" or key=="all":
        from matplotlib import pyplot as plt
        encod_block = PositionalEncoding(d_model=48, max_len=96)
        pe = encod_block.pe.squeeze().T.cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
        pos = ax.imshow(pe, cmap="RdGy", extent=(1,pe.shape[1]+1,pe.shape[0]+1,1))
        fig.colorbar(pos, ax=ax)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Hidden dimension")
        ax.set_title("Positional encoding over hidden dimensions")
        ax.set_xticks([1]+[i*10 for i in range(1,1+pe.shape[1]//10)])
        ax.set_yticks([1]+[i*10 for i in range(1,1+pe.shape[0]//10)])
        plt.show()

        fig, ax = plt.subplots(2, 2, figsize=(12,4))
        ax = [a for a_list in ax for a in a_list]
        for i in range(len(ax)):
            ax[i].plot(np.arange(1,17), pe[i,:16], color=f'C{i}', marker="o", markersize=6, markeredgecolor="black")
            ax[i].set_title(f"Encoding in hidden dimension {i}")
            ax[i].set_xlabel("Position in sequence", fontsize=10)
            ax[i].set_ylabel("Positional encoding", fontsize=10)
            ax[i].set_xticks(np.arange(1,17))
            ax[i].tick_params(axis='both', which='major', labelsize=10)
            ax[i].tick_params(axis='both', which='minor', labelsize=8)
            ax[i].set_ylim(-1.2, 1.2)
        fig.subplots_adjust(hspace=0.8)
        #sns.reset_orig()
        plt.show()

    if key == "causalmask" or key=="all":
        B,L=1,10
        masking = TriangularCausalMask(B,L)
        mask = masking.mask
        print("mask:",mask.shape)
        print(mask)
        batch = torch.rand(B,L,1)
        print("batch:",batch,batch.shape)
        print(batch*mask)
