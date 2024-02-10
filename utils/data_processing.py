import os
import pandas as pd
import glob
import numpy as np

def get_column_data_from_folder(folder_path):
    """
    指定されたフォルダ内のCSVファイルから、テーブル名、列名、カンマ区切りの値を取得します。
    
    Args:
        folder_path (str): CSVファイルが格納されているフォルダのパス
    
    Returns:
        np.array: 取得したデータの配列
    """
    path = os.path.join(folder_path, "*.csv")
    print(path)
    files = glob.glob(path)
    column_data = []
    for file in files:
        table_column_values = split_columns_from_csv(file)
        column_data += table_column_values
    return np.array(column_data)

def split_columns_from_csv(file_path):
    """
    CSVファイルから各列のデータを読み込み、テーブル名、列名、カンマ区切りの値のタプルのリストを返します。
    
    Args:
        file_path (str): 読み込むCSVファイルのパス
    
    Returns:
        list of tuple: (テーブル名, 列名, カンマ区切りの値)のリスト
    """
    df = pd.read_csv(file_path)
    table_name = os.path.basename(file_path).split('.')[0]

    output = []
    for column in df.columns:
        values = df[column].dropna().astype(str).tolist()
        comma_separated_values = ", ".join(values)
        output.append((table_name, column, comma_separated_values))
    return output

def generate_text_from_data(table_name, column_name, values):
    """
    指定されたテーブル名、列名、値のリストから、特定の形式のテキストを生成します。
    
    Args:
        table_name (str): テーブル名
        column_name (str): 列名
        values (list of str): 値のリスト
    
    Returns:
        str: 生成されたテキスト
    """
    max_length = max(len(value) for value in values)
    min_length = min(len(value) for value in values)
    mean_length = np.mean([len(value) for value in values])

    text = f"{table_name}.{column_name} contains {len(values)} values (max: {max_length}, min: {min_length}, mean: {round(mean_length, 1)}): {', '.join(values)}."
    return text

def generate_texts_from_folder(folder_path):
    """
    指定されたフォルダ内のCSVファイルからテキストのリストを生成します。
    
    Args:
        folder_path (str): CSVファイルが格納されているフォルダのパス
    
    Returns:
        list of str: 生成されたテキストのリスト
    """
    data_list = get_column_data_from_folder(folder_path)
    texts = []
    for table_name, column_name, column_values in data_list:
        split_values = column_values.split(", ")
        text = generate_text_from_data(table_name, column_name, split_values)
        texts.append(text)
    return texts

def setup_texts_from_folder(path):
    data_list = get_column_data_from_folder(path)
    texts = []
    for data in data_list:
        table_name, column_name, column_values = data
        splited_column_values = column_values.split(",")
        texts.append(generate_texts_from_folder(table_name, column_name, splited_column_values))
    return texts