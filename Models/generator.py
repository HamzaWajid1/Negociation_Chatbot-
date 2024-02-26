from Models.autoencoder import Transformer
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class Generator(nn.Module):
    def __init__(
            self,
            bert_model_name,
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
            max_length=100            

    ):
        super(Generator,self).__init__()
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_name)
        self.model=BertModel.from_pretrained(bert_model_name)
        self.Transformer=Transformer(src_vocab_size,trg_vocab_size,embed_size,num_layers,forward_expansion,num_heads,dropout,device,max_length)
        self.device=device
    

    def forward(self,input_sentence,output_sentence):
        encoded_dict = self.tokenizer(input_sentence,return_tensors='pt')
        encoder_embedding=self.model(**encoded_dict).last_hidden_state#[:,-1,:]   
        decoded_dict=self.tokenizer(output_sentence,return_tensors='pt')     
        decoder_embedding=self.model(**decoded_dict).last_hidden_state#[:,-1,:]
        output=self.Transformer(encoder_embedding,decoder_embedding)        
        return output
        