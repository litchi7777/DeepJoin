import torch
import random
import argparse

from transformers import MPNetModel, MPNetTokenizer

from models.loss import *
from models.setup import *
from utils.data_processing import *
from utils.dataloader import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the MPNet model")
    parser.add_argument('--batch_size', type=int, default=100, help='Input batch size for training (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    
    args = parser.parse_args()
    return args

def data_aug(splited_column_values):
    """
    与えられた列の値のリストにデータ拡張を行います。
    Args:
        splited_column_values (list of str): カンマで区切られた列の値のリスト
    
    Returns:
        list of str: 拡張された列の値のリスト
    """
    splited_column_values_aug = splited_column_values.copy()
    random.shuffle(splited_column_values_aug)

    #TODO : 閾値を超えて類似する別の列をデータ拡張の一つとして用いる。（参考：論文4.1 Training Data）
    # シャッフルのみのデータ拡張でも普通に動くには動く。

    return splited_column_values_aug

def process_batch(model, tokenizer, data, optimizer=None):
    textsX, textsY = [], []
    table_names, column_names, column_values = data

    for table_name, column_name, column_value in zip(table_names, column_names, column_values):
        splited_column_values = column_value.split(",")
        splited_column_values_aug = splited_column_values.copy()
        textX = generate_text_from_data(table_name, column_name, splited_column_values)
        textY = generate_text_from_data(table_name, column_name, splited_column_values_aug)
        textsX.append(textX)
        textsY.append(textY)

    inputsX = tokenizer(textsX, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputsY = tokenizer(textsY, return_tensors="pt", padding=True, truncation=True, max_length=128)

    outputsX = model(**inputsX)
    last_hidden_statesX = outputsX.last_hidden_state
    outputsY = model(**inputsY)
    last_hidden_statesY = outputsY.last_hidden_state

    cls_token_vectorX = last_hidden_statesX[:, 0, :]
    cls_token_vectorY = last_hidden_statesY[:, 0, :]

    loss = multiple_negative_ranking_loss(cls_token_vectorX, cls_token_vectorY)

    if optimizer:
        loss.backward()  # 逆伝播を行い勾配を計算
        optimizer.step()  # パラメータを更新
        optimizer.zero_grad()  # 勾配をリセット

    return loss.item()

def train(model, tokenizer, train_loader, val_loader, optimizer, epochs=100, patience=10, min_delta=0.001):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs): 
        total_loss = 0
        for data in train_loader:
            loss = process_batch(model, tokenizer, data, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {avg_loss}")
        
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                loss = process_batch(model, tokenizer, data)
                total_loss += loss
            avg_loss = total_loss / len(val_loader)
            print(f"Epoch {epoch}, Validation Loss: {avg_loss}")
            
        # Early Stoppingのチェック
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            patience_counter = 0  # 改善があった場合はカウンタをリセット
            # モデルとトークナイザーの保存
            model.save_pretrained('./result/MPNet/model_ver1.0')
            tokenizer.save_pretrained('./result/MPNet/model_ver1.0')
        else:
            patience_counter += 1  # 改善がなかった場合はカウンタを増やす
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break  # Early Stopping条件を満たした場合はトレーニングを停止

def test(model, tokenizer, test_loader):
    model.eval()  # モデルを評価モードに設定
    total_loss = 0
    with torch.no_grad():  # 勾配計算を無効化
        for data in test_loader:
            loss = process_batch(model, tokenizer, data)
            total_loss += loss
        avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")

def __main__():
    args = parse_args()
    
    train_loader, val_loader, test_loader = setup_dataloaders(batch_size=args.batch_size)
    model, tokenizer, optimizer = setup_model(lr=args.lr)
    train(model, tokenizer, train_loader, val_loader, optimizer, epochs=args.epochs, patience=args.patience, min_delta=args.min_delta)
    test(model, tokenizer, test_loader)

__main__()
