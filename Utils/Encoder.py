import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from Utils.data_utils import LinearModel,PositionalEncoding1,TransformerBlock







class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 num_heads,
                 device,
                 foward_expansion,
                 dropout,
                 max_length 
                ):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device
        #self.input_embedding=nn.Embedding(src_vocab_size,embed_size)
        self.input_embedding=LinearModel(src_vocab_size,embed_size)
        self.positional_encoding=PositionalEncoding1(max_length,embed_size)

        self.layers=nn.ModuleList(
            [
                TransformerBlock(embed_size,num_heads,dropout,foward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,mask):
        batch_size,seq_len=x.shape[1],x.shape[2]
        print("Batch size is : "+str(batch_size))
        print("Seq length is : "+str(seq_len)),
        embedding=self.input_embedding(x)
        #print("input embedding is :\n\n\n "+str(embedding)),
        print("\n\n\nInput embedding shape is :\n\n\n "+str(embedding.shape)),
#        positions=torch.arange(0,seq_len).expand(batch_size,seq_len).to(device=self.device)
        #print("Position is :\n\n\n "+str(positions)),
        #print("\n\n\nPosition shape is :\n\n\n "+str(positions.shape)),
        # Generate positional encoding dynamically based on the input sequence length
        positional_encoding = self.positional_encoding(embedding)
        #positional_encoding = positional_encoding.unsqueeze(0).expand(1,batch_size, embed_size, embed_size)
        #print("Positional encoding is :\n\n\n "+str(positional_encoding)),
        print("\n\n\nPositional encoding shape is :\n\n\n "+str(positional_encoding.shape)),
        
        #out=self.dropout(self.input_embedding(x)+self.positional_encoding(positions))
        out=self.dropout(embedding+positional_encoding)

        for layer in self.layers:
            out=layer(out,out,out,mask)
        return out 