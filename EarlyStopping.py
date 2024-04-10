import os
import torch
import datetime as dt

class EarlyStopping():
    def __init__(self, patience = 10):
        self.patience = patience
        self.model_path = os.path.join(os.getcwd(), "models")
        self.best_loss = float('inf')
        self.counter = 0
    
    def check(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                print("======= Early Stopping ========")
                print(f"Best valid loss: {self.best_loss:.4f}")
                if not os.path.exists(self.model_path):
                    os.makedirs(self.model_path)
                
                file_nm = os.path.join(self.model_path, f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_RNN")
                torch.save(model.state_dict(), file_nm)

                return True
            
            return False
