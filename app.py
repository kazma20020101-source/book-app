import io
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yaml


CONFIG_PATH = Path("config.yml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def list_data_files(data_dir: Path, allowed_exts: List[str]) -> List[Path]:
    if not data_dir.exists():
        return []
    files = []
    for p in data_dir.iterdir():
        if p.is_file() and p.suffix.lower() in allowed_exts:
            files.append(p)
    return sorted(files)


def read_table(path: Path, encoding: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding=encoding, dtype=str)
    return pd.read_excel(path, dtype=str)


def pick_default_mapping(columns: List[str], defaults: Dict[str, List[str]]) -> Dict[str, str]:
    col_set = {c: c for c in columns}
    mapping = {}
    for field, candidates in defaults.items():
        selected = ""
        for cand in candidates:
            if cand in col_set:
                selected = cand
                break
        mapping[field] = selected
    return mapping


def normalize_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for std_field, src_col in mapping.items():
        if src_col and src_col in df.columns:
            out[std_field] = df[src_col].astype(str)
        else:
            out[std_field] = ""
    return out


def build_raw_text(df: pd.DataFrame, fields: List[str]) -> pd.Series:
    combined = df[fields].fillna("").astype(str).agg(" ".join, axis=1)
    return combined


st.set_page_config(page_title="考古学研究室OPAC", layout="wide")

config = load_config()
app_cfg = config.get("app", {})
auth_cfg = config.get("auth", {})
data_cfg = config.get("data", {})
search_cfg = config.get("search", {})
schema_cfg = config.get("schema", {})

st.title(app_cfg.get("title", "考古学研究室OPAC"))
st.caption(app_cfg.get("description", "名古屋大学考古学研究室の蔵書検索・論文検索アプリです。"))

# --- Mode selection ---
mode = st.radio(
    "検索モード",
    ["図書・紀要検索", "論文検索"],
    horizontal=True,
)

# --- Authentication ---
auth_enabled = bool(auth_cfg.get("enabled", False))
if auth_enabled:
    pw_key = auth_cfg.get("password_key", "APP_PASSWORD")
    expected = st.secrets.get(pw_key) or os.getenv(pw_key)
    if not expected:
        st.error("パスワードが設定されていません。`.streamlit/secrets.toml` に設定してください。")
        st.stop()
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        pw = st.text_input("パスワード", type="password")
        if pw and pw == expected:
            st.session_state.auth_ok = True
            st.success("認証に成功しました。")
        elif pw:
            st.error("パスワードが違います。")
        st.stop()

# --- Data selection ---
allowed_exts = data_cfg.get("allowed_extensions", [".csv", ".xlsx", ".xls"])
if mode == "論文検索":
    data_dir = Path(data_cfg.get("directory_papers", "data/papers"))
else:
    data_dir = Path(data_cfg.get("directory_books", "data/books"))
files = list_data_files(data_dir, allowed_exts)

if not files:
    st.warning("選択中のフォルダにCSVまたはExcelファイルを置いてください。")
    st.stop()

selected_file = st.selectbox("データファイルを選択", files, format_func=lambda p: p.name)

# --- Load data ---
@st.cache_data(show_spinner=False)
def load_raw_df(path: Path, encoding: str) -> pd.DataFrame:
    return read_table(path, encoding)

raw_df = load_raw_df(selected_file, data_cfg.get("encoding", "utf-8"))

# --- Column mapping ---
std_fields = schema_cfg.get("fields", [])
if not std_fields:
    std_fields = raw_df.columns.tolist()
default_mapping = pick_default_mapping(raw_df.columns.tolist(), schema_cfg.get("default_column_mapping", {}))

mapping = {}
for field in std_fields:
    default = default_mapping.get(field, "")
    key = f"map_{field}"
    if key not in st.session_state:
        st.session_state[key] = default
    mapping[field] = st.session_state[key]

norm_df = normalize_df(raw_df, mapping)

# --- Search inputs ---
with st.form("search_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        keyword = st.text_input("キーワード")
        title_q = st.text_input("書名")
        author_q = st.text_input("著者")
    with c2:
        year_from = st.text_input("出版年（開始）", placeholder="半角数字で入力")
        year_to = st.text_input("出版年（終了）", placeholder="半角数字で入力")
        publisher_q = st.text_input("出版社")
    with c3:
        sort_key = st.selectbox(
            "並び替え",
            ["出版年（新しい順）", "出版年（古い順）", "書名（昇順）"],
        )
    submitted = st.form_submit_button("検索")

# --- Search logic ---
search_df = norm_df.copy()

# Build raw text for full-text search
raw_text = build_raw_text(norm_df, std_fields)
search_df["raw_text"] = raw_text

if submitted or any([keyword, title_q, author_q, year_from, year_to, publisher_q]):
    if keyword:
        tokens = [t for t in keyword.strip().split() if t]
        if tokens:
            mask = np.ones(len(search_df), dtype=bool)
            lower_text = search_df["raw_text"].str.lower()
            for t in tokens:
                mask &= lower_text.str.contains(t.lower(), na=False)
            search_df = search_df[mask]

    def apply_contains(df: pd.DataFrame, field: str, value: str) -> pd.DataFrame:
        if value:
            return df[df[field].str.contains(value, case=False, na=False)]
        return df

    search_df = apply_contains(search_df, "title", title_q)
    search_df = apply_contains(search_df, "author", author_q)
    search_df = apply_contains(search_df, "publisher", publisher_q)
    if year_from or year_to:
        year_num = pd.to_numeric(search_df["year"], errors="coerce")
        if year_from:
            search_df = search_df[year_num >= pd.to_numeric(year_from, errors="coerce")]
        if year_to:
            search_df = search_df[year_num <= pd.to_numeric(year_to, errors="coerce")]

    if sort_key == "出版年（新しい順）":
        search_df = search_df.assign(_year=pd.to_numeric(search_df["year"], errors="coerce"))
        search_df = search_df.sort_values("_year", ascending=False).drop(columns=["_year"])
    elif sort_key == "出版年（古い順）":
        search_df = search_df.assign(_year=pd.to_numeric(search_df["year"], errors="coerce"))
        search_df = search_df.sort_values("_year", ascending=True).drop(columns=["_year"])
    else:
        search_df = search_df.sort_values("title", ascending=True)

limit = int(search_cfg.get("default_limit", 200))

result_df = search_df.head(limit).drop(columns=["raw_text"], errors="ignore")

text_data = None
buffer = None
if not result_df.empty:
    buffer = io.BytesIO()
    result_df.to_excel(buffer, index=False)
    buffer.seek(0)

    if mode == "図書・紀要検索":
        left_bracket, right_bracket = "『", "』"
    else:
        left_bracket, right_bracket = "「", "」"

    text_lines = []
    for _, row in result_df.iterrows():
        author = str(row.get("author", "") or "").strip()
        year = str(row.get("year", "") or "").strip()
        title = str(row.get("title", "") or "").strip()
        line = f"{author}、{year}、{left_bracket}{title}{right_bracket}"
        text_lines.append(line)
    text_data = "\n".join(text_lines).encode("utf-8")

header_col, btn1_col, btn2_col = st.columns([6, 2, 2])
with header_col:
    st.subheader("検索結果")
    st.write(f"{len(search_df)} 件")
with btn1_col:
    if text_data is not None:
        st.download_button(
            "テキストでダウンロード",
            data=text_data,
            file_name="search_results.txt",
            mime="text/plain",
        )
with btn2_col:
    if buffer is not None:
        st.download_button(
            "Excelでダウンロード",
            data=buffer,
            file_name="search_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.dataframe(result_df, use_container_width=True, height=420)

# --- Copyable text (author,year,title) ---
if text_data is not None:
    st.caption("コピー用テキスト（author,year,「title」）")
    st.text_area(
        "コピー用",
        value=text_data.decode("utf-8"),
        height=160,
    )

# --- Column mapping (below results) ---
with st.expander("列マッピング（基本的には触らない）", expanded=False):
    for field in std_fields:
        cols = [""] + raw_df.columns.tolist()
        current = st.session_state.get(f"map_{field}", "")
        st.selectbox(
            f"{field} に対応する列",
            cols,
            index=cols.index(current) if current in cols else 0,
            key=f"map_{field}",
        )
