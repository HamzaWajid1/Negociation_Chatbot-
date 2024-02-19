
import torch.nn as nn
from Utils.Encoder import Encoder
from Utils.Decoder import decoder





class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            #src_pad_index,
            #trg_pad_index,
            embed_size=64,
            num_layers=6,
            forward_expansion=4,
            num_heads=8,
            dropout=0,
            device="cuda",
            max_length=100,
            enc_mask=None,
            dec_mask=1
            

    ):
        super(Transformer,self).__init__()
        self.encoder=Encoder(src_vocab_size,embed_size,num_layers,num_heads,device,forward_expansion,dropout,max_length)
        self.decoder=decoder(trg_vocab_size,embed_size,num_heads,num_layers,forward_expansion,dropout,device,max_length)
        #self.src_pad_index=src_pad_index
        #self.trg_pad_index=trg_pad_index
        self.device=device
    

    def forward(self,src,trg):
        enc_src=self.encoder(src,None)
        print("\n\n\n\nThe shape of encoder is "+str(enc_src.shape))
        out=self.decoder(trg,enc_src,None,1)
        print("\n\n\n\nThe shape of output is "+str(out.shape))
        return out