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

TITLE = "産業構造 × 事業所密度・雇用密度（全国比較）"
CAPTION = "e-Stat 経済センサス × 国勢調査（人口1万人あたり指標）"

# ======================
# CSS Injection
# ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

    /* Global Settings */
    .stApp {
        font-family: 'Noto Sans JP', "Hiragino Kaku Gothic ProN", "Hiragino Sans", Meiryo, sans-serif;
    }
    
    /* Header modernization */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        color: var(--text-color) !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--secondary-background-color);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-color) !important;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 600;
        font-size: 1.8rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--secondary-background-color);
    }
    
    /* Sidebar Text - ensure it uses the main text color variable */
    section[data-testid="stSidebar"] * {
        color: var(--text-color);
    }
    
    /* Specific overrides for sidebar inputs/texts to ensure visibility */
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] input {
        color: var(--text-color) !important;
    }

    /* Fix for dropdown menu items */
    div[data-baseweb="popover"] ul li span {
        color: var(--text-color) !important; 
    }
    div[data-baseweb="select"] div {
         color: var(--text-color) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: var(--text-color);
        font-weight: 500;
        opacity: 0.7;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color) !important;
        border-bottom-color: var(--primary-color) !important;
        opacity: 1;
    }
    /* Card Styling */
    .css-card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
</style>
""", unsafe_allow_html=True)

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
]

JP_RENAME = {
    "areaName": "地域名",
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
    データフィルタリング：
    1. 全国(00000)除外
    2. 人口ゼロ除外
    3. 県全体の集計行(XX000)を除外
    4. 政令指定都市の重複除外（「市全体」と「区」が両方ある場合、「市全体」を除外）
    """
    d = df.copy()

    # 全国(00000)除外
    d = d[d[AREA_COL] != "00000"]
    # 人口ゼロ除外
    d = d[d["population"] > 0]
    # 県集計(XX000)を除外
    d = d[~d[AREA_COL].str.endswith("000")]

    if pref_code != "00":
        d = d[d[AREA_COL].str.startswith(pref_code)].copy()

    # ---------------------------
    # 重複除外ロジック (Double Counting Fix - Strict)
    # ---------------------------
    # 政令指定都市などは「市全体(XXXX0)」と「区(XXXX1~)」の両方が入っている場合がある。
    # 区が存在するなら市全体は除外する。
    # XXXX0 というコードだけでなく、XXXX0 のような「親」コード全般をチェック。
    # ロジック：「末尾が0」かつ「政令指定都市のパターン（3桁目が1）」かつ「自分を除いた前方一致（4桁）するコードが存在する」なら除外。
    # 市部（Mobara 12210 等）は3桁目が2なので除外されないようにする。
    
    all_codes = set(d[AREA_COL].unique())
    remove_codes = set()
    
    for code in all_codes:
        # 末尾が0で、かつ県全体(000)ではない
        # かつ、政令指定都市（xx1xxのパターン、例14100）であること。
        # 通常の市（xx2xx、例12210）は除外しない。
        if code.endswith("0") and not code.endswith("000"):
            # 3桁目が '1' かどうかチェック (index 2)
            # コードは string, 5桁保証 (zfill済み)
            is_designated = (len(code) == 5 and code[2] == '1')
            
            if is_designated:
                # プレフィックス (先頭4桁)
                prefix = code[:-1] 
                
                # 同じプレフィックスを持つ他のコードが存在するか？
                has_children = d[
                    (d[AREA_COL].str.startswith(prefix)) & 
                    (d[AREA_COL] != code)
                ].shape[0] > 0
                
                if has_children:
                    remove_codes.add(code)
    
    if remove_codes:
        d = d[~d[AREA_COL].isin(remove_codes)]

    return d


def apply_industry(d: pd.DataFrame, sic_code: str) -> pd.DataFrame:
    """
    産業を適用。総計なら市区町村×年次で合算。
    """
    if sic_code == TOTAL_CODE:
        # 人口は合算せず、代表値（max/first）をとる
        # 従業者・事業所は合算
        out = (
            d.groupby([AREA_COL, "areaName", "@time"], as_index=False)
            .agg({
                "establishments": "sum",
                "employees": "sum",
                "population": "max" # 同じ地域なら人口は同じはずなのでmaxでよい
            })
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
    view = df.loc[:, DISPLAY_COLS + ["est_dev", "emp_dev"]].rename(columns=JP_RENAME).copy()

    # 見出しを短縮（2行で収まりよく）
    view = view.rename(
        columns={
            "事業所数": "事業所",
            "従業者数": "従業者",
            "事業所密度（人口1万人あたり）": "事業所\n密度",
            "雇用密度（人口1万人あたり）": "雇用\n密度",
            "est_dev": "事業所\n(県差)",
            "emp_dev": "雇用\n(県差)",
        }
    )

    # 念のため数値化
    for c in ["事業所", "従業者", "人口"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce")

    return view.style.format(
        {
            "事業所": "{:,.0f}",
            "従業者": "{:,.0f}",
            "人口": "{:,.0f}",
            "事業所\n密度": "{:,.0f}",
            "雇用\n密度": "{:,.0f}",
            "事業所\n(県差)": "{:+,.0f}",
            "雇用\n(県差)": "{:+,.0f}",
        },
        na_rep="—",
    )


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
            alt.Tooltip("est_dev:Q", title="事業所密度(県差)", format="+,.0f"),
            alt.Tooltip("emp_dev:Q", title="雇用密度(県差)", format="+,.0f"),
        ],
    )

    points = base.mark_circle(size=80, opacity=0.7).encode(
        color=alt.value("#3182ce"),  # Modern Blue
        stroke=alt.value("white"),
        strokeWidth=alt.value(1)
    )

    layers = [points]

    # 県平均ライン（ある場合のみ）
    if est_avg is not None:
        vline = alt.Chart(pd.DataFrame({"x": [est_avg]})).mark_rule(
            strokeDash=[4, 4], 
            color="#e53e3e",  # Red for average
            strokeWidth=2
        ).encode(x="x:Q")
        layers.append(vline)

    if emp_avg is not None:
        hline = alt.Chart(pd.DataFrame({"y": [emp_avg]})).mark_rule(
            strokeDash=[4, 4], 
            color="#e53e3e",
            strokeWidth=2
        ).encode(y="y:Q")
        layers.append(hline)

    chart = alt.layer(*layers).properties(height=550).configure_view(
        strokeWidth=0
    ).configure_axis(
        titleFontWeight="bold"
    ).interactive()
    return chart


# ======================
# UI
# ======================
# タイトルを1行に収めるためのCSS調整
st.markdown(f"""
<h1 style='font-size: 1.8rem; margin-bottom: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
    {TITLE}
</h1>
<p style='color: #718096; margin-top: 0;'>{CAPTION}</p>
""", unsafe_allow_html=True)

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

# use_dev_sort = st.sidebar.checkbox("ランキングを『県平均との差』で並べ替える", value=True)
# ↑ 削除し、デフォルトの指標順（降順）にする

population_min = st.sidebar.slider("人口下限（ノイズ抑制）", 0, 20000, 5000, step=500)
topn = st.sidebar.slider("表示件数（ランキング）", 10, 200, 50)

# 1) スコープ（全国/県）→ 市区町村
scope_df = filter_scope_base(base, pref_code=pref_code)

# 2) 産業適用（総計なら合算）
d_all = apply_industry(scope_df, sic_code=sic_code)

# 3) ノイズ抑制（人口下限）
d = d_all[d_all["population"] >= population_min].copy()

# 4) 県平均（人口加重）→ 県平均との差
# Calculate averages on the full dataset (d_all) for accurate Reference metrics, 
# OR keep it based on filtered 'd'?
# Usually, reference average should include everything (so d_all), 
# but the current logic was using `d`.
# BUT user wants "Total Population" to be Japan Total (126M).
# That sum comes from `avg['pop_sum']`. 
# So we need to calculate `avg` from `d_all` OR verify where `avg` comes from.
# Currently: `avg = compute_weighted_avg(d)` -> d is filtered.
# FIX: Use `d_all` for calculating the Total Population metric and averages.
avg = compute_weighted_avg(d_all) 
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

    # デフォルトソート: 指標（密度）の降順のみ
    sort_col = metric_col
    
    rank = (
        d.sort_values(sort_col, ascending=False)
        .head(topn)
        .reset_index(drop=True)
    )
    rank.insert(0, "順位", rank.index + 1)

    st.dataframe(
        format_table(rank),
        use_container_width=True,
        hide_index=True,
        height=600,  # Fixed height for single page view
        column_config={
            "地域名": st.column_config.TextColumn(width="medium"),
            "事業所": st.column_config.NumberColumn(width="small"),
            "従業者": st.column_config.NumberColumn(width="small"),
            "人口": st.column_config.NumberColumn(width="small"),
            # 密度の列も一応small指定でコンパクトに
        }
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


# ======================
# サイトの目的・定義（About）
# ======================
st.divider()

st.markdown("""
<div class="css-card">
<h2 style='margin-top: 0; color: #3182ce;'>Industrial Density Dashboard について</h2>
<p style='font-size: 1.1rem; color: #4a5568;'>産業構造 × 事業所密度・雇用密度（全国比較）Streamlitダッシュボード</p>
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;'>
<div>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>1. 開発の背景（Why）</h3>
<p>地域の産業政策や診断では「どの産業が集積しているか」を掴む必要がありますが、実数だけでは人口規模に引きずられ、適切な比較が困難です。本ツールは人口1万人あたりの<b>「密度」</b>に着目し、規模を正規化した比較を可能にします。</p>
</div>
<div>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>2. 意思決定への活用（So what）</h3>
<ul>
<li><b>自治体政策：</b>重点産業の選定、集積構造の把握</li>
<li><b>調査・コンサル：</b>初期診断・見取り図の作成</li>
<li><b>企業誘致：</b>産業の性格（零細型か大規模拠点型か）の判別</li>
</ul>
</div>
</div>
<div style='margin-top: 1.5rem;'>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>3. 分析の視点（What）</h3>
<p>以下の4つの論点を通じて、地域の産業集積を可視化します。</p>
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
<div style='background: rgba(49, 130, 206, 0.1); padding: 10px; border-radius: 8px;'>
<b>A: 事業所の厚み</b><br>密度が高い自治体はどこか
</div>
<div style='background: rgba(49, 130, 206, 0.1); padding: 10px; border-radius: 8px;'>
<b>B: 雇用の厚み</b><br>雇用力が強い自治体はどこか
</div>
<div style='background: rgba(49, 130, 206, 0.1); padding: 10px; border-radius: 8px;'>
<b>C: 両指標のズレ</b><br>産業の性格差（零細 vs 大規模）
</div>
<div style='background: rgba(49, 130, 206, 0.1); padding: 10px; border-radius: 8px;'>
<b>D: 県平均との差</b><br>県内での相対的なポジション
</div>
</div>
</div>
<div style='margin-top: 1.5rem;'>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>4. 主な機能（Features）</h3>
<ul>
<li><b>柔軟なフィルタ：</b>都道府県、産業大分類、人口下限（ノイズ抑制）による絞り込み</li>
<li><b>多角的な可視化：</b>ランキングでの詳細比較と、散布図による構造把握</li>
<li><b>県基準の指標：</b>人口加重平均による「県平均ライン」を自動算出</li>
</ul>
</div>
<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;'>
<div>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>5. 指標の定義（Evidence）</h3>
<ul style='font-family: monospace; font-size: 0.9rem; list-style: none; padding-left: 0;'>
<li>・事業所密度 = 事業所数 / 人口 × 10,000</li>
<li>・雇用密度 = 従業者数 / 人口 × 10,000</li>
<li>・県平均 = (Σ実数 / Σ人口) × 10,000</li>
<li>・県差 = 自治体密度 - 県平均</li>
</ul>
</div>
<div>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>6. データ仕様（Data & Logic）</h3>
<p style='font-size: 0.9rem;'>
<b>出典：</b>e-Stat 経済センサス（2014-） / 国勢調査（2020）<br>
<b>基礎処理：</b>全国行除外、人口ゼロ除外、県集計行（XX000）除外等のクレンジングを実施済。
</p>
</div>
</div>
<div style='margin-top: 1.5rem;'>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>7. 前処理ロジック（重要な設計判断）</h3>
<div style='background: rgba(128, 128, 128, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.1);'>
<p><b>7.1 集計の混入を防ぐ（ベースのフィルタ）</b><br>
filter_scope_base() にて以下を実施し、比較対象を「市区町村の実体」に揃える。<br>
・全国行（area == "00000"）を除外<br>
・population > 0 のみ採用<br>
・県集計行（XX000）を除外</p>
<p><b>7.2 政令指定都市の二重計上対策</b><br>
市全体（末尾0）と区（末尾1〜）が両方ある場合、市全体を除外（二重計上防止）。<br>
条件：末尾0 かつ 3桁目が1（例：14100）で、子要素（区）が存在する場合。</p>
<p><b>7.3 総計（全産業）の扱い</b><br>
「全産業」選択時、自治体×年次で再集計。密度を正確に再計算し整合性を担保。</p>
</div>
</div>
<div style='margin-top: 1.5rem;'>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>8. 結果（Findings：この画面で分かること）</h3>
<ul>
<li>市区町村別に、産業ごとの<b>集積（事業所密度）</b>と<b>雇用の厚み（雇用密度）</b>を同時に比較可能。</li>
<li>県平均（人口加重）を基準に、各自治体の<b>県内での相対ポジション（県差）</b>を説明可能。</li>
<li>事業所密度×雇用密度の組み合わせにより「零細中心」か「大規模拠点型」かの構造差を推定。</li>
</ul>
</div>
<div style='margin-top: 1.5rem;'>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>9. 示唆（Implications：打ち手案トップ3）</h3>
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;'>
<div style='background: rgba(49, 130, 206, 0.1); padding: 15px; border-radius: 10px;'>
<b>優先度A：重点候補の仮置き</b><br>密度で当たりを付け、売上・付加価値・生産性など「稼ぐ力」指標へ接続して絞り込む。
</div>
<div style='background: rgba(49, 130, 206, 0.1); padding: 15px; border-radius: 10px;'>
<b>優先度B：施策タイプの切り替え</b><br>零細中心ならDX・共同化、雇用が厚いなら人材供給・住環境など、打ち手を最適化。
</div>
<div style='background: rgba(49, 130, 206, 0.1); padding: 15px; border-radius: 10px;'>
<b>優先度C：ベンチマーク</b><br>散布図で似た位置にある自治体を特定し、政策や支援スキームの参照先にする。
</div>
</div>
</div>
<div style='margin-top: 1.5rem;'>
<h3 style='border-left: 4px solid #3182ce; padding-left: 10px;'>10. 再現性（How：セットアップ & 実行）</h3>
<div style='background: rgba(0, 0, 0, 0.1); padding: 15px; border-radius: 12px; font-family: monospace; font-size: 0.85rem;'>
# ディレクトリ構成<br>
app.py / requirements.txt / data/base_2014_ec_2020_pop_level2.parquet<br><br>
# セットアップ & 実行<br>
pip install -r requirements.txt<br>
streamlit run app.py
</div>
</div>
</div>
""", unsafe_allow_html=True)
