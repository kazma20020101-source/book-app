# 考古学研究室リテラボ蔵書アプリ

Streamlit を用いた蔵書検索アプリです。CSV/Excel の蔵書データを読み込み、全文検索・項目別検索、検索結果のダウンロードを提供します。

## 使い方

1) 依存関係をインストール

```bash
pip install -r requirements.txt
```

2) データを配置

`data/` に CSV または Excel を配置します。

3) 起動

```bash
streamlit run app.py
```

## 設定

`config.yml` でタイトルや OpenAI モデル、検索件数などを調整できます。

## データの列名マッピング

アプリ内の「列マッピング」から、既存データの列名を標準フィールドに対応付けできます。

標準フィールド: id, title, author, year, publisher, location, call_number, keywords, notes

## メモ

パスワード機能は `config.yml` の `auth.enabled` で切り替えます。
