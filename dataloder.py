import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# 時系列データをwindowに分割
def devide_by_window(config, data, label):
    out_seq = []
    L = len(data)
    window = config['data']['window']
    for i in range(L - window):
        seq = torch.FloatTensor(data[i: i+window])
        seq_label = torch.FloatTensor(label[i: i+window])
        out_seq.append((seq, seq_label))
    
    return out_seq

# dataloaderを作成
def make_dataloader(config):
    config_data = config['data']
    df = pd.read_csv("https://raw.githubusercontent.com/aweglteo/tokyo_weather_data/main/data.csv", parse_dates=True, index_col=0)


    data = df[['cloud', 'wind', 'ave_tmp', 'max_tmp', 'min_tmp', 'rain']].values[:-1]
    label = df['ave_tmp'].values[1:]



    seq = devide_by_window(config, data, label)

    train_size = int(len(df) * config_data['train_size'])
    valid_size = int(len(df) * config_data['valid_size'])
    train_dataset = seq[:train_size]
    valid_dataset = seq[train_size: train_size+valid_size]
    test_dataset  = seq[train_size+valid_size: ]

    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=config_data['batch_size'], shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config_data['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config_data['batch_size'], shuffle=False)

    return train_loader, valid_loader, test_loader, label