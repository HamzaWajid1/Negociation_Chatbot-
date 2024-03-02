import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from Utils.data_utils import LinearModel,PositionalEncoding1,Decoder_block










class decoder(nn.Module):
    def __init__(
            self,
            target_vocab_size,
            embed_size,
            num_heads,
            num_layers,
            forward_expansion,
            dropout,
            device,
            max_length

    ):
        super(decoder,self).__init__()
        self.device=device
        #self.word_embedding=nn.Embedding(target_vocab_size,embed_size)
        self.word_embedding=LinearModel(target_vocab_size,embed_size)
        self.positional_encoding=PositionalEncoding1(max_length,embed_size)

        self.layers=nn.ModuleList(
            [
                Decoder_block(embed_size,num_heads,forward_expansion,dropout,device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out=nn.Linear(embed_size,target_vocab_size)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,enc_out,src_mask,trg_mask):
        input_embedding=self.word_embedding(x)
        positional_encoding=self.positional_encoding(input_embedding)

        x=self.dropout(input_embedding+positional_encoding)

        for layer in self.layers:
            x=layer(x,enc_out,enc_out,src_mask,trg_mask)

            out=self.fc_out(x)
        return out