from transformers import MPNetModel, MPNetTokenizer
import faiss
import torch
import numpy as np
from utils.data_processing import *  # データ処理用のユーティリティ関数をインポート
from models.setup import *  # モデルセットアップ関数をインポート

def find_nearest_text(query_text, model, tokenizer, index):
    """
    与えられたクエリテキストに最も近いテキストを検索します。

    Args:
        query_text (str): 検索クエリとなるテキスト
        model (transformers.PreTrainedModel): トークン化とベクトル化を行うための事前学習済みモデル
        tokenizer (transformers.PreTrainedTokenizer): テキストをトークン化するためのトークナイザー
        index (faiss.Index): 検索を行うためのFAISSインデックス

    Returns:
        tuple: (最も近いテキストとの距離, 最も近いテキストのインデックス)
    """
    # クエリテキストをトークン化し、モデルが受け入れられる形式に変換
    inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # トークン化されたテキストをモデルに入力し、最終隠れ状態を取得
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    # CLSトークンのベクトルを取得し、Numpy配列に変換
    cls_token_vector = last_hidden_states[:, 0, :].detach().cpu().numpy()
    cls_token_vector = np.ascontiguousarray(cls_token_vector)
    
    # FAISSインデックスを使って最も近いテキストを検索
    D, I = index.search(cls_token_vector, 1)
    D = D[0][0]  # 最も近いテキストとの距離
    I = I[0][0]  # 最も近いテキストのインデックス
    return D, I

def __main__():
    # modelとtokenizerのロード
    model_path = "./result/MPNet/model_ver1.0"
    model, tokenizer = load_model(model_path)
    
    # FAISS indexのロード
    index_path = "./result/faiss/sample.index"
    index = faiss.read_index(index_path)
    
    # リポジトリ内のテキストとクエリテキストを設定
    repository_texts = setup_texts_from_folder("./datasets/raw/train")
    query_texts = setup_texts_from_folder("./datasets/raw/test")
    query_text = query_texts[0]  # 最初のクエリテキストを使用

    # クエリテキストに最も近いテキストを検索
    d, i = find_nearest_text(query_text, model, tokenizer, index)
    
    # 結果を表示
    print(f"Query: {query_text}")
    print(f"Result: {repository_texts[i]}")

# メイン関数を実行
__main__()
