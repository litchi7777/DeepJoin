from transformers import MPNetModel, MPNetTokenizer
import torch

def load_model(model_path):
    """
    指定されたパスからMPNetモデルとトークナイザーをロードします。
    
    Args:
        model_path (str): モデルとトークナイザーが保存されているディレクトリのパス
    
    Returns:
        tuple: ロードされたMPNetモデルとトークナイザー
    """
    # 指定されたパスからモデルとトークナイザーをロード
    model = MPNetModel.from_pretrained(model_path)
    tokenizer = MPNetTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def setup_model(lr=0.001):
    """
    新しいMPNetモデルとトークナイザーをロードし、オプティマイザーを設定します。
    
    Args:
        lr (float, optional): オプティマイザーの学習率。デフォルトは0.001。
    
    Returns:
        tuple: ロードされたMPNetモデル、トークナイザー、そして設定されたオプティマイザー
    """
    # 'microsoft/mpnet-base'からモデルとトークナイザーをロード
    model = MPNetModel.from_pretrained('microsoft/mpnet-base')
    tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')
    
    # Adamオプティマイザーを使用してモデルのパラメータを最適化するOptimizerを作成
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return model, tokenizer, optimizer