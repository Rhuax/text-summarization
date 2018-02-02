



import torch
from torch.nn import Module,Dropout,Embedding,GRU,Linear

import torch.nn.functional as F
from util import *
from torch.nn import Parameter

class AttnDecoderRNN(Module):
    def __init__(self,hidden_size,output_size,embeddings,dropout_p=0.1,max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size=hidden_size
        self.embedding_size=embeddings.size(1)
        self.output_size=output_size
        self.dropout_p=dropout_p
        self.max_length=max_length
        self.embedding=Embedding(self.output_size,self.embedding_size)
        self.embedding.weight = Parameter(embeddings)

        self.attn=Linear(self.hidden_size+self.embedding_size,self.max_length)
        self.attn_combine=Linear(self.hidden_size+self.embedding_size,self.hidden_size)
        self.dropout=Dropout(self.dropout_p)
        self.gru=GRU(self.hidden_size,self.hidden_size)
        self.out=Linear(self.hidden_size,self.output_size)

    def forward(self,input,hidden,encoder_outputs):
        embedded=self.embedding(input)
        embedded=embedded.view(1,1,-1)
        embedded=self.dropout(embedded)

        cat_temp=torch.cat((embedded[0],hidden[0].double()),1)
        calc_att=self.attn(cat_temp.float())
        attn_weights=F.softmax(calc_att,dim=1)
        attn_applied=torch.bmm(attn_weights.unsqueeze(0).double(),encoder_outputs.unsqueeze(0).double()).float()

        output=torch.cat((embedded[0].float(),attn_applied[0]),1)
        output=self.attn_combine(output).float().unsqueeze(0)

        output=F.relu(output)

        output,hidden=self.gru(output,hidden)

        output=F.log_softmax(self.out(output[0]),dim=1)
        return output,hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result.cuda()
