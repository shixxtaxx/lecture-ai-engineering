import time
import pytest
import pandas as pd
import pickle
import os
import numpy as np
from pathlib import Path


@pytest.fixture
def model_path():
    return Path("day5/演習3/models/titanic_model.pkl")


@pytest.fixture
def test_data():
    # テスト用のサンプルデータを作成
    data_path = Path("day5/演習3/data/Titanic.csv")
    df = pd.read_csv(data_path)
    # 前処理（モデルの学習時と同じ形式にする）
    X = df.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1)

    # 数値型カラムと文字列型カラムを分けて処理
    numeric_cols = X.select_dtypes(include=["number"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    # 文字列型カラムは最頻値で埋める
    categorical_cols = X.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    return X.head(100)  # テスト用に100サンプルを使用


def test_model_inference_time(model_path, test_data):
    """モデルの推論時間が許容範囲内であることを検証"""
    # モデルの読み込み
    assert os.path.exists(model_path), f"モデルファイルが存在しません: {model_path}"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 推論時間の計測
    start_time = time.time()
    predictions = model.predict(test_data)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論結果の検証
    assert len(predictions) == len(
        test_data
    ), "予測結果のサンプル数が入力と一致しません"
    assert all(
        isinstance(pred, (int, np.int64))
        or (isinstance(pred, float) and pred.is_integer())
        for pred in predictions
    ), "予測結果は整数値である必要があります"

    # 推論時間の検証（100サンプルで0.5秒以内を期待）
    assert (
        inference_time < 0.5
    ), f"推論時間が許容範囲を超えています: {inference_time:.4f}秒"

    print(f"推論時間: {inference_time:.4f}秒（{len(test_data)}サンプル）")
