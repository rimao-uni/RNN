import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config_model = config['model']
        self.input_size  = self.config_model['input_size'] #4
        self.hidden_size = self.config_model['hidden_size'] #8
        self.output_size = self.config_model['output_size'] #1

        self.input_layer  = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.act    = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config_model['dropout_rate'])
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 隠れ状態の初期化
        h = self.init_hidden(batch_size).to(x.device)

        for t in range(seq_len):
            h_new = self.act(self.input_layer(x[:, t, :]))   # 入力を隠れ状態に変換
            combined = torch.cat((h_new, h), dim=1)          # 隠れ状態と前の隠れ状態を結合
            combined = self.dropout(combined)
            h = self.act(self.hidden_layer(combined))        # 結合したものを隠れ層に入力
        
        out = self.output_layer(h)                 # 出力

        return out
    
    def init_hidden(self, batch_size):
        # 初期隠れ状態をゼロで初期化する
        return torch.zeros(batch_size, self.hidden_size)
