# DeepJoin
DeepJoinの実装のための簡単なサンプルです。

## How to use
1. データセットのセットアップ
- 実際はデータベースを用いた実装を行うが，サンプルにはtrain, test, valにcsv形式のテーブルを最低限追加している。

2. MPNetのFinetuning
- MPNetのFinetuningはfinetuning.pyによって行う。
- 適宜，パラメータやモデル名などを変更する必要があります。

3. Offline phase
- Offline phaseは offline.pyを用いて行う。
- Offline phaseではANN Searchの一つであるfaissのindex作成を行う。

4. Online phase
- Online phaseは offline.pyを用いて行う。
- Online phaseでは作成されたFaiss indexを用いてqueryと最も類似する列を探索する。