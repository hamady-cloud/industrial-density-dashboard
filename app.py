import streamlit as st
import pandas as pd
import altair as alt


# ======================
# 設定
# ======================
st.set_page_config(
    page_title="産業構造 × 事業所・雇用密度（全国比較）",
    layout="wide",
)


# ======================
# Custom CSS Injection
# ======================
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #3E404D;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            text-align: center;
        }
        [data-testid="stMetricLabel"] {
            color: #9CA3AF;
            font-size: 0.9rem;
        }
        [data-testid="stMetricValue"] {
            color: #FAFAFA;
            font-weight: 700;
        }

        /* Header Styling */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 700;
            color: #FAFAFA;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #262730;
            border-radius: 5px;
        }

        /* DataFrame Styling */
        .stDataFrame {
            border: 1px solid #3E404D;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

TITLE = "産業構造 × 事業所密度・雇用密度（全国比較）"
CAPTION = "e-Stat 経済センサス × 国勢調査（人口1万人あたり指標）"

DATA_PATH = "data/base_2014_ec_2020_pop_level2.parquet"

AREA_COL = "area"
SIC_COL = "sicCode"
TOTAL_CODE = "__TOTAL__"
TOTAL_NAME = "総計（全産業）"

METRIC_OPTIONS = {
    "事業所密度": "est_density",
    "雇用密度": "emp_density",
}

DISPLAY_COLS = [
    "areaName",
    "establishments",
    "employees",
    "population",
    "est_density",
    "emp_density",
    "est_rank",
    "emp_rank",
]

JP_RENAME = {
    "areaName": "地域名",
    "est_rank": "全国順位（事業所）",
    "emp_rank": "全国順位（雇用）",
    "establishments": "事業所数",
    "employees": "従業者数",
    "population": "人口",
    "est_density": "事業所密度（人口1万人あたり）",
    "emp_density": "雇用密度（人口1万人あたり）",
}


# ======================
# データ読み込み
# ======================
@st.cache_data(show_spinner=False)
def load_base(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df[AREA_COL] = df[AREA_COL].astype(str).str.zfill(5)
    # level=2: 030/2700/7330 など桁が混在するため zfill しない
    df[SIC_COL] = df[SIC_COL].astype(str).str.strip()
    df["pref"] = df[AREA_COL].str[:2]
    return df


def build_pref_maps(df: pd.DataFrame):
    pref_name_map = (
        df[df[AREA_COL].str.endswith("000") & (df[AREA_COL] != "00000")]
        .drop_duplicates("pref")
        .set_index("pref")["areaName"]
        .to_dict()
    )
    pref_list = sorted(p for p in df["pref"].unique() if p != "00")
    return pref_list, pref_name_map


def build_sic_lists(df: pd.DataFrame):
    sic_df = (
        df[[SIC_COL, "sicName"]]
        .dropna(subset=[SIC_COL, "sicName"])
        .drop_duplicates()
        .copy()
    )
    sic_df[SIC_COL] = sic_df[SIC_COL].astype(str).str.strip()
    sic_df["sicName"] = sic_df["sicName"].astype(str).str.strip()

    sic_df["__num"] = pd.to_numeric(sic_df[SIC_COL], errors="coerce")
    sic_df = sic_df.sort_values(["__num", SIC_COL]).drop(columns="__num").reset_index(drop=True)

    total_row = pd.DataFrame([{SIC_COL: TOTAL_CODE, "sicName": TOTAL_NAME}])
    use = pd.concat([total_row, sic_df], ignore_index=True)

    sic_codes = use[SIC_COL].tolist()
    sic_map = use.set_index(SIC_COL)["sicName"].to_dict()
    return sic_codes, sic_map, 0  # 総計をデフォルト


def filter_scope_base(df: pd.DataFrame, pref_code: str) -> pd.DataFrame:
    """
    スコープ（全国/都道府県）と、市区町村のみに整形
    """
    d = df.copy()

    # 全国(00000)除外
    d = d[d[AREA_COL] != "00000"]
    # 人口ゼロ除外
    d = d[d["population"] > 0]
    # 市区町村のみ（都道府県 XX000 を除外）
    d = d[~d[AREA_COL].str.endswith("000")]

    # 政令指定都市の「市全体」（XX100, XX130, XX150等）を除外（区 XX1XX があるため重複する）
    # 区を持つ政令指定都市は、XX100などのコードで合算値を持っているが、
    # 区ごとのデータも持っているため、単純合算すると倍になる。
    # ここでは、第3桁が '1' で、かつ第4,5桁が '00', '30', '50' のものを除外するヒューリスティックを入れる
    # 横浜市(14100), 川崎市(14130), 相模原市(14150)などが該当
    # 一般的な市は XX201 ～ なので、XX1XX は政令市の特徴。
    # ただし XX100 以外に XX130 (川崎), XX150 (相模原), XX140 (堺) などがある。
    # 安全策として「末尾が0」かつ「第3桁が1」を除外（区は XX101...XX1xx で末尾が0でないことが多いが、10区などは0になるかも？いや区コードは連番）
    # 区コードは通常 01, 02... なので XX110 (10番目の区) はありえる。
    # したがって、「政令市全体コード」の特定リストを除外するか、
    # あるいは「第3桁が1」かつ「末尾が00, 30, 40, 50」などを除外。
    
    # データを確認すると:
    # 14100 (横浜), 14130 (川崎), 14150 (相模原)
    # これらは下2桁が 00, 30, 50.
    # 一方、区は 01, 02 ... 18 (都筑区).
    # 区コードで末尾が 0 になるのは、XX110 (10番目の区), XX120 (20番目の区)...
    # 横浜市戸塚区(14110) は末尾0だが、これは除外してはいけない！
    
    # したがって、除外すべきは 「政令市の全体コード」 のみ。
    # 既知のパターン: XX100 (多くの政令市), XX130 (川崎, 北九州, 福岡?), XX140 (堺, 浜松?), XX150 (相模原, 熊本?)
    # 安全なロジック:
    # 「区（XX1XX）」が存在する市（XX）の、「市全体行」を除去する。
    # 実装: 
    # 1. 第3桁が1 ("..1..") の行を抽出
    # 2. その中で、重複カウントの原因となる「親」コードを除く。
    # 親コードのルール： 末尾が '00' (XX100) はほぼ確実に親。
    # 川崎(14130), 相模原(14150) は特殊。
    
    # 今回のデータセット特有の重複排除
    ignore_suffixes = ["100", "130", "140", "150"]
    d = d[~d[AREA_COL].str.endswith(tuple(ignore_suffixes))]

    if pref_code != "00":
        d = d[d[AREA_COL].str.startswith(pref_code)].copy()

    return d


def apply_industry(d: pd.DataFrame, sic_code: str) -> pd.DataFrame:
    """
    産業を適用。総計なら市区町村×年次で合算。
    """
    if sic_code == TOTAL_CODE:
        out = (
            d.groupby([AREA_COL, "areaName", "@time"], as_index=False)
            .agg(
                {
                    "establishments": "sum",
                    "employees": "sum",
                    "population": "max",  # Population is constant per area, take max (or first)
                }
            )
        )
        out["sicName"] = TOTAL_NAME
        out[SIC_COL] = TOTAL_CODE
        out["est_density"] = out["establishments"] / out["population"] * 10000
        out["emp_density"] = out["employees"] / out["population"] * 10000
        return out

    return d[d[SIC_COL] == str(sic_code)].copy()


def compute_weighted_avg(d: pd.DataFrame) -> dict:
    """
    人口加重平均（=県全体を1つの自治体としてみなした密度）
    """
    pop_sum = float(pd.to_numeric(d["population"], errors="coerce").sum())
    est_sum = float(pd.to_numeric(d["establishments"], errors="coerce").sum())
    emp_sum = float(pd.to_numeric(d["employees"], errors="coerce").sum())

    if pop_sum <= 0:
        return {"pop_sum": 0.0, "est_avg": None, "emp_avg": None}

    est_avg = est_sum / pop_sum * 10000
    emp_avg = emp_sum / pop_sum * 10000
    return {"pop_sum": pop_sum, "est_avg": est_avg, "emp_avg": emp_avg}


def add_deviation_cols(d: pd.DataFrame, est_avg: float | None, emp_avg: float | None) -> pd.DataFrame:
    out = d.copy()
    out["est_dev"] = out["est_density"] - est_avg if est_avg is not None else None
    out["emp_dev"] = out["emp_density"] - emp_avg if emp_avg is not None else None
    return out



def format_table(df: pd.DataFrame):
    """
    データフレームを表示用に整形（数値化と列名整理のみ）
    styleオブジェクトではなく、素のDataFrameを返す（st.column_configで装飾するため）
    """
    view = df.loc[:, DISPLAY_COLS + ["est_dev", "emp_dev"]].copy()

    # 念のため数値化。NaNは0にして整数型へ変換（カンマ区切りのデフォルト適用のため）
    for c in ["establishments", "employees", "population", "est_density", "emp_density", "est_dev", "emp_dev", "est_rank", "emp_rank"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0).astype(int)
    
    return view


def get_column_config(df: pd.DataFrame):
    """
    カラムごとの表示設定（プログレスバーなど）を生成
    """
    # 密度の最大値を取得（バーのスケール用）
    max_est_density = df["est_density"].max() if not df["est_density"].empty else 100
    max_emp_density = df["emp_density"].max() if not df["emp_density"].empty else 100

    return {
        "areaName": st.column_config.TextColumn("地域名", width="medium", pinned=True),
        "population": st.column_config.NumberColumn("人口", help="国勢調査人口"),
        "establishments": st.column_config.NumberColumn("事業所数"),
        "employees": st.column_config.NumberColumn("従業者数"),
        
        "est_density": st.column_config.ProgressColumn(
            "事業所密度",
            help="人口1万人あたり",
            format="%d",
            min_value=0,
            max_value=max_est_density,
        ),
        "emp_density": st.column_config.ProgressColumn(
            "雇用密度",
            help="人口1万人あたり",
            format="%d",
            min_value=0,
            max_value=max_emp_density,
        ),
        
        "est_dev": st.column_config.NumberColumn(
            "事業所密度（差）",
            help="県平均との差",
            format="%+d",
        ),
        "emp_dev": st.column_config.NumberColumn(
            "雇用密度（差）",
            help="県平均との差",
            format="%+d",
        ),
        "est_rank": st.column_config.NumberColumn(
            "全国順位（事業所）",
            help="全国の市区町村中の順位（事業所密度）",
        ),
        "emp_rank": st.column_config.NumberColumn(
            "全国順位（雇用）",
            help="全国の市区町村中の順位（雇用密度）",
        ),
    }


def make_scatter(d: pd.DataFrame, est_avg: float | None, emp_avg: float | None):
    base = alt.Chart(d).encode(
        x=alt.X("est_density:Q", title="事業所密度（人口1万人あたり）"),
        y=alt.Y("emp_density:Q", title="雇用密度（人口1万人あたり）"),
        size=alt.Size("population:Q", title="人口"),
        tooltip=[
            alt.Tooltip("areaName:N", title="地域名"),
            alt.Tooltip("population:Q", title="人口", format=",.0f"),
            alt.Tooltip("est_density:Q", title="事業所密度", format=",.0f"),
            alt.Tooltip("emp_density:Q", title="雇用密度", format=",.0f"),
            alt.Tooltip("est_dev:Q", title="事業所密度(県平均との差)", format="+,.0f"),
            alt.Tooltip("emp_dev:Q", title="雇用密度(県平均との差)", format="+,.0f"),
        ],
    )

    points = base.mark_circle(opacity=0.6)

    layers = [points]

    # 県平均ライン（ある場合のみ）
    if est_avg is not None:
        vline = alt.Chart(pd.DataFrame({"x": [est_avg]})).mark_rule(strokeDash=[6, 4], color="#FAFAFA").encode(x="x:Q")
        layers.append(vline)

    if emp_avg is not None:
        hline = alt.Chart(pd.DataFrame({"y": [emp_avg]})).mark_rule(strokeDash=[6, 4], color="#FAFAFA").encode(y="y:Q")
        layers.append(hline)

    chart = (
        alt.layer(*layers)
        .properties(height=520)
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            gridColor="#3E404D",
            domainColor="#3E404D",
            tickColor="#3E404D",
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )
    return chart


# ======================
# UI
# ======================
st.title(TITLE)
st.caption(CAPTION)

base = load_base(DATA_PATH)

pref_list, pref_name_map = build_pref_maps(base)
sic_codes, sic_map, default_sic_index = build_sic_lists(base)

st.sidebar.header("表示条件")

pref_code = st.sidebar.selectbox(
    "都道府県",
    options=["00"] + pref_list,
    format_func=lambda p: "全国" if p == "00" else f"{p}：{pref_name_map.get(p, '')}",
)

sic_code = st.sidebar.selectbox(
    "産業（大分類）",
    options=sic_codes,
    index=default_sic_index,
    # 番号は表示しない（名称だけ）
    format_func=lambda c: sic_map.get(c, ""),
)

metric_label = st.sidebar.radio("指標", list(METRIC_OPTIONS.keys()))
metric_col = METRIC_OPTIONS[metric_label]

population_min = st.sidebar.slider("人口下限（ノイズ抑制）", 0, 20000, 5000, step=500)
topn = st.sidebar.slider("表示件数（ランキング）", 10, 200, 50)

# 1) まず「全国」ベースで市区町村を抽出（順位計算のため）
#    pref_code="00" で呼び出すと全国全ての市区町村が返る
nat_scope_df = filter_scope_base(base, pref_code="00")

# 2) 産業適用（総計なら合算）
#    これを全国データに対して行う
d_all_nat = apply_industry(nat_scope_df, sic_code=sic_code)

# 3) ノイズ抑制（人口下限）
#    これもランキング計算前に適用（対象外はランキング外とする想定）
d_nat = d_all_nat[d_all_nat["population"] >= population_min].copy()

# 4) 全国順位の計算
d_nat["est_rank"] = d_nat["est_density"].rank(ascending=False, method="min")
d_nat["emp_rank"] = d_nat["emp_density"].rank(ascending=False, method="min")

# 5) 都道府県フィルタ（表示用）
if pref_code != "00":
    d = d_nat[d_nat[AREA_COL].str.startswith(pref_code)].copy()
else:
    d = d_nat.copy()

# 4) 県平均（人口加重）→ 県平均との差
avg = compute_weighted_avg(d)
est_avg = avg["est_avg"]
emp_avg = avg["emp_avg"]

d = add_deviation_cols(d, est_avg=est_avg, emp_avg=emp_avg)

# ヘッダ：いま見ているスコープ
scope_name = "全国" if pref_code == "00" else pref_name_map.get(pref_code, pref_code)
sic_name = sic_map.get(sic_code, "")

st.markdown(f"#### スコープ：**{scope_name}**　｜　産業：**{sic_name}**　｜　人口下限：**{population_min:,} 人**")

# 県平均の表示（カード）
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("対象人口（合計）", f"{avg['pop_sum']:,.0f}")
with c2:
    st.metric("県平均 事業所密度", "—" if est_avg is None else f"{est_avg:,.0f}")
with c3:
    st.metric("県平均 雇用密度", "—" if emp_avg is None else f"{emp_avg:,.0f}")

tab1, tab2 = st.tabs(["ランキング", "散布図（県平均ライン）"])

# ======================
# ① ランキング
# ======================
with tab1:
    st.subheader(f"ランキング（{scope_name}）")

    # シンプルに選択した指標（密度）でソート
    sort_col = metric_col

    rank = (
        d.sort_values(sort_col, ascending=False)
        .head(topn)
        .reset_index(drop=True)
    )
    rank.insert(0, "順位", rank.index + 1)

    rank_df = format_table(rank)
    st.dataframe(
        rank_df,
        column_config=get_column_config(rank_df),
        use_container_width=True,
        hide_index=True,
        height=500,
    )



# ======================
# ② 散布図（県平均ライン）
# ======================
with tab2:
    st.subheader("事業所密度 × 雇用密度（県平均ライン付き）")
    st.caption("破線：県平均（人口加重平均）｜ 点サイズ：人口（人口下限後）")

    scatter_df = d.dropna(subset=["est_density", "emp_density", "population"])
    chart = make_scatter(scatter_df, est_avg=est_avg, emp_avg=emp_avg)
    st.altair_chart(chart, use_container_width=True)
