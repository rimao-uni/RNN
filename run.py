import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(model, dataloader, criterion, optim, state):
    running_loss = 0.0
    pred_list = []
    if state == 'train':
        model.train()             
        for x, answer in dataloader:
            x = x.to(device)  
            optim.zero_grad()   
            pred = model(x)     
            pred_list += list(pred)

            loss = criterion(pred, answer)          
            loss.backward()              
            optim.step()                             
            running_loss += loss.item() * x.size(0)
    
    elif state == 'eval':
        model.eval()                               
        with torch.no_grad():                           
            for x, answer in dataloader:
                x = x.to(device)
                pred = model(x)
                pred_list += list(pred)
                loss = criterion(pred, answer)
                running_loss += loss.item() * x.size(0)
    
    else:
        raise Exception("予想外の引数")
    
    return running_loss / len(dataloader), pred_list