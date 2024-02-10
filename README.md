# DeepJoin
DeepJoinの実装のための簡単なサンプル。

## How to use
### データセットのセットアップ
　実際はデータベースを用いた実装を行いますが，サンプルにはtrain, test, valにcsv形式のテーブルを最低限追加している。

### MPNetのFinetuning
　MPNetのFinetuningはfinetuning.pyを用いる。
パラメータやモデル名などを指定する必要がある。

デフォルトのデータでは `python finetuning --batch_size 2 --epochs 3`などとして動作を確認できる。

### Offline phase
　Offline phaseは offline.pyを用いる。

Offline phaseではANN Searchの一つであるfaissのindex作成する。

### Online phase
　Online phaseは offline.pyを用いる。

Online phaseでは作成されたFaiss indexを用いてqueryと最も類似する列を探索する。

## TODO
- データベースとの接続
- 論文内で記述されているシャッフル以外のpositive sampleの作成方法の追加
