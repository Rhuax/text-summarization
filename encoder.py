from torch.nn import Module,Embedding,GRU
from torch.autograd import Variable
import torch.nn
from torch.nn import Parameter

class EncoderRNN(Module):
    def __init__(self,input_size, hidden_size,pre_trained_embedding):
        super(EncoderRNN, self).__init__()
        self.hidden_size=hidden_size
        self.embedding=Embedding(input_size,pre_trained_embedding.size(1))
        self.embedding.weight=Parameter(pre_trained_embedding)
        self.gru=GRU(pre_trained_embedding.size(1),hidden_size)

    def forward(self, input, hidden):
        embedded=self.embedding(input)
        embedded=embedded.view(1,1,-1)
        output=embedded
        output,hidden=self.gru(output,hidden)
        return output,hidden
    def initHidden(self):
        result=Variable(torch.zeros(1,1,self.hidden_size))
        return result.cuda()