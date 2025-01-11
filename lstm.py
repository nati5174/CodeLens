import torch
from torch import nn



## general structure for LSTM model, will be adding to it
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_layers, num_stack_layers):
    super().__init__()
    self.hidden_layers = hidden_layers
    self.num_stack_layers = num_stack_layers

    self.lstm = nn.LSTM(input_size, hidden_layers, num_stack_layers, batch_first=True)

    #self.fc = nn. # any other layers we will
    #....
    #....
    #....
    #....


  def forward(self, x):
    batch_size = x.size(0)

    #hiden state
    h0 = torch.zeros(self.num_stack_layers, batch_size, self.hidden_layers) #hiden state (stacked layers, bath size, hidden layers input)
    c0 = torch.zeros(self.num_stack_layers, batch_size, self.hidden_layers) #cell state (stacked layers, bath size, hidden layers input

    #pass
    out,_ = self.lstm(x, (h0, c0)) #out will shape, (batch_size, sequence_length, hidden_size)
                                   # _ --> The final hiden states and cell states it will store

    out = self.fc(out[:, -1, :])


    return out   
