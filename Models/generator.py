from Models.autoencoder import Transformer
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import BertConfig

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
            max_length=100,
            vocab_size: int = 30522,
            hidden_size: int = 768, 
            num_hidden_layers: int = 12, 
            num_attention_heads: int = 12, 
            intermediate_size: int = 3072, 
            hidden_act: str = "gelu", 
            hidden_dropout_prob: float = 0.1, 
            attention_probs_dropout_prob: float = 0.1, 
            max_position_embeddings: int = 512, 
            type_vocab_size: int = 2, 
            initializer_range: float = 0.02, 
            layer_norm_eps: float = 1e-12, 
            pad_token_id: int = 0, 
            position_embedding_type: str = "absolute", 
            use_cache: bool = True            

    ):
        super(Generator,self).__init__()
        self.tokenizer=BertTokenizer.from_pretrained(bert_model_name)
        self.model=BertModel.from_pretrained(bert_model_name)
        #self.Transformer=Transformer(src_vocab_size,trg_vocab_size,embed_size,num_layers,forward_expansion,num_heads,dropout,device,max_length)
        self.Transformer=BertModel(BertConfig(vocab_size=vocab_size,num_hidden_layers=num_hidden_layers,hidden_size=hidden_size,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size,hidden_act=hidden_act,hidden_dropout_prob=hidden_dropout_prob,attention_probs_dropout_prob=attention_probs_dropout_prob,max_position_embeddings=max_position_embeddings,type_vocab_size=type_vocab_size,initializer_range=initializer_range,layer_norm_eps=layer_norm_eps,pad_token_id=pad_token_id,position_embedding_type=position_embedding_type,use_cache=use_cache))
        self.device=device
    

    def forward(self,input_sentence,output_sentence):
        encoded_dict = self.tokenizer(input_sentence,return_tensors='pt')
        encoder_embedding=self.model(**encoded_dict).last_hidden_state#[:,-1,:]   
        decoded_dict=self.tokenizer(output_sentence,return_tensors='pt')     
        decoder_embedding=self.model(**decoded_dict).last_hidden_state#[:,-1,:]
        output=self.Transformer(encoder_embedding,decoder_embedding)        
        return output
        