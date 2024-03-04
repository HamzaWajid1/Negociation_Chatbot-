from Utils.Encoder import Encoder
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel


class Discriminator(nn.Module):
    def __init__(
            self,
            bert_model_name,
            src_vocab_size,          
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
        super(Discriminator,self).__init__()
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_name)
        self.model=BertModel.from_pretrained(bert_model_name)
        self.Encoder=Encoder(src_vocab_size,embed_size,num_layers,num_heads,device,forward_expansion,dropout,max_length)
        self.device=device
    

    def forward(self,question_sentence,fake_real_sentence):
        question_dict = self.tokenizer(question_sentence,return_tensors='pt')
        #encoder_embedding=self.model(**encoded_dict).last_hidden_state#[:,-1,:] 
        fake_real_dict=self.tokenizer(fake_real_sentence,return_tensors='pt')     
        #decoder_embedding=self.model(**decoded_dict).last_hidden_state#[:,-1,:]
        #concat = question_dict + fake_real_dict
        # Concatenate tokenized embeddings
        concat_input = {key: torch.cat([question_dict[key], fake_real_dict[key]], dim=1) for key in question_dict}

        # Convert dictionary of tensors to a single tensor
        concatenated_tensor = torch.cat(list(concat_input.values()), dim=2)
        output=self.Encoder(concatenated_tensor,None)        
        return output