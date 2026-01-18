import streamlit as st
import pandas as pd


# ======================
# 設定
# ======================
st.set_page_config(
    page_title="産業構造 × 事業所・雇用密度（全国比較）",
    layout="wide",
)

TITLE = "産業構造 × 事業所密度・雇用密度（全国比較）"
CAPTION = "e-Stat 経済センサス × 国勢調査（人口1万人あたり指標）"

DATA_PATH = "data/base_2009_ec_2020_pop.parquet"  # ←必要なら base_2014... に変更

AREA_COL = "area"
SIC_COL = "sicCode"

METRIC_OPTIONS = {
    "事業所密度（人口1万人あたり）": "est_density",
    "雇用密度（人口1万人あたり）": "emp_density",
}

JP_RENAME = {
    "areaName": "地域名",
    "area": "地域コード",
    "establishments": "事業所数",
    "employees": "従業者数",
    "population": "人口",
    "est_density": "事業所密度（人口1万人あたり）",
    "emp_density": "雇用密度（人口1万人あたり）",
}


# ======================
# Data loading & prep
# ======================
@st.cache_data(show_spinner=False)
def load_base(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def preprocess_base(base: pd.DataFrame) -> pd.DataFrame:
    """
    必要な型・派生列を整える
    - area: 5桁ゼロ埋め
    - sicCode: 文字列化（ゼロ埋めはしない：A〜T対応のため）
    - pref: 先頭2桁
    """
    df = base.copy()

    df[AREA_COL] = df[AREA_COL].astype(str).str.zfill(5)
    df[SIC_COL] = df[SIC_COL].astype(str)
    df["pref"] = df[AREA_COL].str[:2]

    return df


def build_pref_maps(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """
    pref_list: '01','02',..（全国00除外）
    pref_name_map: '01'-> '北海道' みたいな辞書（XX000のareaNameから作る）
    """
    pref_name_map = (
        df[df[AREA_COL].str.endswith("000") & (df[AREA_COL] != "00000")]
        .drop_duplicates("pref")
        .set_index("pref")["areaName"]
        .to_dict()
    )
    pref_list = sorted(p for p in df["pref"].unique() if p != "00")
    return pref_list, pref_name_map


def build_sic_lists(df: pd.DataFrame) -> tuple[list[str], dict[str, str], int]:
    """
    sic_codes: 選択肢
    sic_map: code->name
    default_index: 既定選択（000があればそこ）
    """
    sic_list = (
        df[[SIC_COL, "sicName"]]
        .drop_duplicates()
        .sort_values(SIC_COL)
        .reset_index(drop=True)
    )
    sic_codes = sic_list[SIC_COL].tolist()
    sic_map = sic_list.set_index(SIC_COL)["sicName"].to_dict()

    default_index = 0
    if "000" in sic_codes:
        default_index = sic_codes.index("000")

    return sic_codes, sic_map, default_index


def filter_base(df: pd.DataFrame, sic_code: str) -> pd.DataFrame:
    """
    共通フィルタ：
    - 産業コード一致
    - 全国(00000)除外
    - 人口ゼロ除外
    """
    out = df[df[SIC_COL] == sic_code].copy()
    out = out[out[AREA_COL] != "00000"]
    out = out[out["population"] > 0]
    return out


def jp_view(df: pd.DataFrame) -> pd.DataFrame:
    """表示用に日本語列名へ変換"""
    return df.rename(columns=JP_RENAME)


# ======================
# UI
# ======================
st.title(TITLE)
st.caption(CAPTION)

base = preprocess_base(load_base(DATA_PATH))

pref_list, pref_name_map = build_pref_maps(base)
sic_codes, sic_map, sic_default_index = build_sic_lists(base)

st.sidebar.header("表示条件")

pref_code = st.sidebar.selectbox(
    "都道府県",
    options=pref_list,
    format_func=lambda p: f"{p}：{pref_name_map.get(p, '')}",
)

sic_code = st.sidebar.selectbox(
    "産業（大分類）",
    options=sic_codes,
    index=sic_default_index,
    format_func=lambda c: f"{c}：{sic_map.get(c, '')}",
)

metric_label = st.sidebar.radio("指標", options=list(METRIC_OPTIONS.keys()))
metric_col = METRIC_OPTIONS[metric_label]

topn = st.sidebar.slider("表示件数（ランキング）", 10, 200, 50)

# 抽出
d = filter_base(base, sic_code)


# ======================
# Tabs
# ======================
tab1, tab2, tab3 = st.tabs(["全国ランキング", "都道府県→市町村", "散布図（政策示唆）"])

with tab1:
    st.subheader("全国ランキング（ベンチマーク）")

    rank = (
        d.sort_values(metric_col, ascending=False)
        .head(topn)
        .loc[:, ["areaName", AREA_COL, "establishments", "employees", "population", "est_density", "emp_density"]]
        .reset_index(drop=True)
    )
    rank.insert(0, "全国順位", rank.index + 1)

    st.dataframe(jp_view(rank), use_container_width=True)

with tab2:
    st.subheader("都道府県内 市町村比較")

    pref_df = d[d[AREA_COL].str.startswith(pref_code)].copy()
    pref_df = pref_df[~pref_df[AREA_COL].str.endswith("000")]  # 市区町村だけ（都道府県XX000は除外）
    pref_df = pref_df.sort_values(metric_col, ascending=False)

    pref_jp = jp_view(pref_df)

    st.dataframe(
        pref_jp[
            [
                "地域名", "地域コード",
                "事業所数", "従業者数", "人口",
                "事業所密度（人口1万人あたり）",
                "雇用密度（人口1万人あたり）",
            ]
        ].rename(columns={"地域名": "市町村名"}),
        use_container_width=True,
    )

with tab3:
    st.subheader("事業所密度 × 雇用密度（政策ターゲティング）")
    st.caption("点の大きさ＝人口")

    scatter_df = d.dropna(subset=["est_density", "emp_density", "population"])
    st.scatter_chart(scatter_df, x="est_density", y="emp_density", size="population")
