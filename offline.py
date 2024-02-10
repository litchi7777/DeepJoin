from transformers import MPNetModel, MPNetTokenizer
import faiss
import torch
from utils.data_processing import *
from models.setup import *

def __main__():
    # リポジトリ内のテキストを設定
    repository_texts = setup_texts_from_folder("./datasets/raw/train")

    # 事前学習済みモデルとトークナイザーをロード
    model_path = "./result/MPNet/model_ver1.0"
    model, tokenizer = load_model(model_path)

    # リポジトリテキストをモデルで処理可能な形式にトークン化
    inputs = tokenizer(repository_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # トークン化されたテキストをモデルに通して、最後の隠れ状態を取得
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    # CLSトークンのベクトルを取得して、Numpy配列に変換
    cls_token_vector = last_hidden_states[:, 0, :].detach().cpu().numpy()

    # 特徴ベクトルの次元を取得し、FAISSインデックスを初期化
    d = cls_token_vector.shape[1]  # 特徴ベクトルの次元を取得
    index = faiss.IndexFlatL2(d)  # L2距離を使ったインデックスを作成

    # トークン化されたテキストのベクトルをFAISSインデックスに追加
    index.add(cls_token_vector)

    # FAISSインデックスをディスクに保存
    faiss.write_index(index, "./result/faiss/sample.index")

# プログラムのエントリーポイント
if __name__ == "__main__":
    __main__()