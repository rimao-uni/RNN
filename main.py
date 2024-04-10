import json
import torch
import torch.nn as nn
import torch.optim as optim

from dataloder import make_dataloader
from models import RNN
from save_loss import save_loss
from run import run
from EarlyStopping import EarlyStopping


if __name__ == '__main__':
    
    # configファイルの読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seedの設定
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])   

    train_loader, valid_loader, test_loader, label = make_dataloader(config)

    # モデル、評価関数、活性化関数の定義
    model = RNN(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['optim']['lr'])
    es = EarlyStopping()

    # 学習
    train_losses = []; valid_losses = []
    for epoch in range(config['epochs']):
        train_loss, _ = run(model, train_loader, criterion, optimizer, state='train')
        valid_loss, _ = run(model, valid_loader, criterion, optimizer, state='eval')
        if epoch % 1 == 0:
            print(f"Epoch: {epoch+1}  train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Early Stopping
        if es.check(valid_loss, model):
            break
    
    # 評価
    test_loss, test_pred = run(model, test_loader, criterion, optimizer, state='eval')
    print(f"test_loss: {test_loss:.4f}")

    # 画像出力
    save_loss(train_losses, valid_losses)
