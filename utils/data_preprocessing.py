import pandas as pd

def generate_text_from_column(table_name, column_name, data):
    """
    指定されたテーブル名、列名、データリストから特定の形式のテキストを生成する関数。
    
    Args:
    - table_name: テーブル名
    - column_name: 列名
    - data: データのリスト
    
    Returns:
    - text: 生成されたテキスト
    """
    # 各データの値の長さを計算
    value_lengths = [len(str(value)) for value in data]
    
    # 最大値、最小値、平均値の長さを計算
    max_length = max(value_lengths)
    min_length = min(value_lengths)
    mean_length = sum(value_lengths) / len(value_lengths)
    
    # テキストを生成
    text = f"{table_name}. {column_name} contains {len(data)} values ({max_length}, {min_length}, {round(mean_length, 1)}): {', '.join(data)}."
    
    return text

def process_csv_columns_to_texts(file_path):
    """
    CSVファイルの各列に対してテキストを生成する関数。
    
    Args:
    - file_path: CSVファイルのパス
    
    Returns:
    - texts: 生成されたテキストのリスト
    """
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    
    # CSV名を取得（ファイル名から拡張子を除去）
    table_name = file_path.split('/')[-1].split('.')[0]
    
    # 各列に対してテキストを生成
    texts = []
    for column in df.columns:
        data = df[column].dropna().astype(str).tolist()  # NaNを除去し、文字列として扱う
        text = generate_text_from_column(table_name, column, data)
        texts.append(text)
    
    return texts