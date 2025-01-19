from torch import nn
class customOCR(nn.Module):

  def __init__(self,num_class):
    super().__init__()
    #input size (5, 3, 128, 1024)
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,kernel_size=3,stride=1, padding=1) #(64, 128, 1024)
    self.relu1 = nn.Relu()
    self.max1 = nn.MaxPool2d(kernel_size=(2,2), stride=2) #output (64, 64, 512)

    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=5,stride=2, padding=2) #(128, 64, 512)
    self.relu2 = nn.Relu()
    self.max2 = nn.MaxPool2d(kernel_size=(2,2), stride=2) #(128, 32, 256)


    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=5,stride=2, padding=2) #(256, 32, 256)
    self.relu3 = nn.Relu()
    self.max3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)#(256, 16, 128)

    self.ft = nn.Flatten()

    self.linear1 = nn.Linear(256*16*128, 1024)
    self.linear2 = nn.Linear(1024, 512)
    self.linear3 = nn.Linear(512, 256)
    self.linear4 = nn.Linear(256, 128)  #(batch_size, 128)

    self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3, bidirectional=True, batch_first=True, dropout=0.3)  #(5, 1, 256)

    self.linear5 = nn.Linear(128*2, num_class) 


  def forward(self, x):
    bs, c, h, w = x.size() 
    x = self.conv1(x)  
    x = self.relu1(x)
    x = self.max1(x)


    x = self.conv2(x)  
    x = self.relu2(x)
    x = self.max2(x)

    x = self.conv2(x)  
    x = self.relu2(x)
    x = self.max2(x)

    x = self.ft(x)

    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    x = self.linear4(x)

    x = x.unsqueeze(1)

    x, _ = self.lstm(x)

    x = self.Linear(x[:,-1,:])

