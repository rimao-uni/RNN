import os
import datetime as dt
import matplotlib.pyplot as plt

def save_loss(train_loss, valid_loss, save_path="log"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title("losses")
    plt.ylabel('loss')
    plt.xlabel("epoch")
    plt.legend()
    

    file_nm = os.path.join(save_path, f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_loss.png")
    plt.savefig(file_nm)
    plt.close()
    plt.show()


