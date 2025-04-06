from sched import scheduler
import torch
import numpy as np
from tqdm import tqdm
from model import EmotionNet
from torch.optim.lr_scheduler import StepLR
from utils import prepare_dataloaders
from config import config
import matplotlib.pyplot as plt

def train():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionNet(num_classes=len(config.classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    
    # 数据加载
    dataloaders = prepare_dataloaders()
    
    # 训练循环
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    patience = 10
    counter = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                
                # 保存最佳模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 
                              f"{config.checkpoint_dir}/best_model.pth")
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping")
                        break
        if counter >= patience:
            break            
                    
    # 保存训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Validation")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.savefig(f"{config.result_dir}/training_curves.png")

if __name__ == "__main__":
    train()