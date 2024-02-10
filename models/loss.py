import torch
import torch.nn.functional as F

def multiple_negative_ranking_loss(X, Y):
    """
    Multiple negative ranking lossを計算します。
    
    Args:
    - X: バッチ内の各サンプルに対する埋め込みベクトルのテンソル。サイズは (N, D) です。
    - Y: バッチ内の各サンプルに対する正のサンプルの埋め込みベクトルのテンソル。サイズは (N, D) です。
    
    Returns:
    - loss: 計算された損失の値。
    """
    # コサイン類似度の計算
    scores = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
    
    # 正のサンプルに対するスコア（対角成分）
    positive_scores = scores.diag()
    
    # ソフトマックス正規化スコアの負の対数尤度
    loss = -torch.log(torch.exp(positive_scores) / torch.sum(torch.exp(scores), dim=1))
    return loss.mean()