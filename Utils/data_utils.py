import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
import math





class attention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(attention, self).__init__()
        self.embed_size=embed_size
        self.num_heads=num_heads
        self.head_dim= embed_size // num_heads

        assert(self.head_dim * num_heads == embed_size), "Embed size must be completely divisible by number of heads"

        self.query=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.key=nn.Linear(self.embed_size,self.head_dim,bias=False)
        self.value=nn.Linear(self.embed_size,self.head_dim,bias=False)
        self.output=nn.Linear(num_heads*self.head_dim,embed_size)

    def forward(self,query,key,value,mask):
        batch_size=query.shape[0]
        q_len,k_len,v_len=query.shape[1],key.shape[1],value.shape[1]
        print("\n\n\nThe shapes before permutetion")
        query=query.reshape(batch_size,q_len,self.num_heads,self.head_dim)
        print("The shape of query is "+str(query.shape))
        key=key.reshape(batch_size,k_len,self.num_heads,self.head_dim)
        print("The shape of key is "+str(key.shape))
        value=value.reshape(batch_size,v_len,self.num_heads,self.head_dim)
        print("The shape of value is "+str(value.shape))
        # Transpose to perform batch matrix multiplication
        query = query.permute(0, 2, 1, 3)  # (batch_size, num_heads, q_len, head_dim)
        key = key.permute(0, 2, 1, 3)  # (batch_size, num_heads, k_len, head_dim)
        value = value.permute(0, 2, 1, 3)  # (batch_size, num_heads, v_len, head_dim)
        print("\n\n\nThe shapes after permutetion")
        print("The shape of query is "+str(query.shape))
        print("The shape of key is "+str(key.shape))
        print("The shape of value is "+str(value.shape))
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply mask if provided
        if mask is not None:
            mask1=torch.full(scores.size(),float('-inf'))
            #print(mask1)
            mask1=torch.triu(mask1,diagonal=1)
            #print(mask1)
            #print(scores)
            scores += mask1
            #print(scores)
        
        # Apply softmax to obtain attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout if needed (you can add this if desired)
        # attention_weights = self.dropout(attention_weights)

        # Apply attention weights to the values
        output = torch.matmul(attention_weights, value)
        print("The initial shape of output is "+str(output.shape))
        # Reshape and concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, q_len, num_heads, head_dim)
        output = output.reshape(batch_size, q_len, self.num_heads * self.head_dim)
        
        # Linear transformation to get the final output
        output = self.output(output)
        print("The final shape of output is "+str(output.shape))
        return output  
    









class TransformerBlock(nn.Module):
    def __init__(self, embed_size,num_heads,dropout,foward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention=attention(embed_size,num_heads=num_heads)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)

        self.feed_foward=nn.Sequential(
            nn.Linear(embed_size,foward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(foward_expansion*embed_size,embed_size)
        )

        self.dropout=nn.Dropout(dropout)

    def forward(self,query,key,value,mask):
        multi_head_attention=self.attention(query,key,value,mask)
        x=self.dropout(self.norm1(multi_head_attention+query))
        foward=self.feed_foward(x)  
        out=self.dropout(self.norm2(foward + x))
        return out
    









class PositionalEncoding1(nn.Module):
    def __init__(self, max_len, embed_size):
        super(PositionalEncoding1, self).__init__()
        self.embed_size = embed_size

        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure that positional encoding has the same shape as input tensor x
        batch_size, seq_len = x.shape[:2]
        pe = self.pe[:, :seq_len, :].expand(batch_size, -1, -1)
        return pe
    




class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()  # Using ReLU activation function

    def forward(self, x):
        # Linear transformation
        linear_output = self.linear(x)
        # Apply activation function
        output = self.activation(linear_output)
        return output  






class Decoder_block(nn.Module):
    def __init__(self,embed_size,num_heads,forward_expansion,dropout,device):
        super(Decoder_block,self).__init__()
        self.multi_head_attention=attention(embed_size,num_heads)
        self.norm=nn.LayerNorm(embed_size)
        self.transformer_block=TransformerBlock(embed_size,num_heads,dropout,forward_expansion)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,key,value,src_mask,targ_mask):
        multi_head_attention=self.multi_head_attention(x,x,x,targ_mask)
        query=self.dropout(self.norm(x+multi_head_attention))
        out=self.transformer_block(query,key,value,src_mask)
        return out
