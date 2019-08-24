import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块

class LSTM_Regression(nn.Module):
    def __init__(self, input_dim, hidden_units_size, output_dim,num_layers=1):
        super(LSTM_Regression, self).__init__()
        self.input_dim = input_dim
        self.hidden_units_size = hidden_units_size
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_units_size,num_layers)        
        self.Linear = nn.Linear(hidden_units_size, output_dim)        

    
    def forward(self, data):
        lstm_out, self.hidden = self.lstm(data)
        s,b,h = lstm_out.shape
        lstm_out = lstm_out.view(s*b,h)#8*1*50
        y_ = self.Linear(lstm_out)
        y_ = y_.view(s,b,-1)
        return y_
    
model = LSTM_Regression(input_dim, hidden_units, output_dim,1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
