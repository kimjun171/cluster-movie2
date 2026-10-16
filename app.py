import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 및 브라우저 탭 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 군집화)")

# 2. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 숫자형 컬럼 변환
    num_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    total_count = len(df)
    
    # 네 속성 중 값이 없거나 first_week_audi가 0인 영화 제외
    valid_mask = (
        df["first_scrn"].notna() & (df["first_scrn"] > 0) &
        df["total_audi"].notna() & (df["total_audi"] > 0) &
        df["days_in_top10"].notna() &
        df["first_week_audi"].notna() & (df["first_week_audi"] > 0)
    )
    filtered_df = df[valid_mask].copy()
    
    # 속성 가공
    # 1) 스크린 수 (상용로그)
    filtered_df["스크린 수 (상용로그)"] = np.log10(filtered_df["first_scrn"])
    # 2) 누적 관객 (상용로그)
    filtered_df["누적 관객 (상용로그)"] = np.log10(filtered_df["total_audi"])
    # 3) 100위권 일수 (그대로)
    filtered_df["10위권 일수"] = filtered_df["days_in_top10"]
    # 4) 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20으로 클리핑)
    filtered_df["롱런 지수"] = (filtered_df["total_audi"] / filtered_df["first_week_audi"]).clip(upper=20)
    
    clustered_count = len(filtered_df)
    return filtered_df, total_count, clustered_count

df, total_count, clustered_count = load_data()

# 전체 편수와 묶은 편수를 화면에 한 줄로 표시
st.write(f"**전체 영화 수:** {total_count:,}편 | **분석 및 묶음에 사용된 영화 수:** {clustered_count:,}편")
st.markdown("---")

# 3. 속성 고르기 (멀티셀렉트)
all_features = ["스크린 수 (상용로그)", "누적 관객 (상용로그)", "10위권 일수", "롱런 지수"]

selected_features = st.multiselect(
    "묶는 데 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=all_features,
    default=all_features
)

if len(selected_features) < 2:
    st.warning("속성을 둘 이상 선택해 주세요.")
    st.stop()

# 4. 표준화 및 K-평균 군집화 (k=3, 난수 고정 random_state=42)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df["raw_cluster"] = raw_clusters

# 5. 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = df.groupby("raw_cluster")["total_audi"].mean().sort_values(ascending=False)
symbol_map = {cluster_id: symbol for cluster_id, symbol in zip(cluster_audi_mean.index, ["㉮", "㉯", "㉰"])}

df["묶음"] = df["raw_cluster"].map(symbol_map)

# 6. 2차원 산점도
st.subheader("📌 2차원 산점도")
col2d_x, col2d_y = st.columns(2)

with col2d_x:
    x_2d = st.selectbox("가로축(X축) 속성 선택", options=selected_features, index=0, key="2d_x")
with col2d_y:
    y_2d = st.selectbox("세로축(Y축) 속성 선택", options=selected_features, index=1 if len(selected_features) > 1 else 0, key="2d_y")

fig_2d = px.scatter(
    df,
    x=x_2d,
    y=y_2d,
    color="묶음",
    hover_name="movieNm",
    category_orders={"묶음": ["㉮", "㉯", "㉰"]},
    color_discrete_map={"㉮": "#FF4B4B", "㉯": "#00C0F2", "㉰": "#00D4B1"},
    title=f"2D 산점도 ({x_2d} vs {y_2d})"
)
st.plotly_chart(fig_2d, use_container_width=True)

# 7. 3차원 산점도
st.subheader("🧊 3차원 산점도")

if len(selected_features) >= 3:
    col3d_x, col3d_y, col3d_z = st.columns(3)
    with col3d_x:
        x_3d = st.selectbox("X축 속성 선택", options=selected_features, index=0, key="3d_x")
    with col3d_y:
        y_3d = st.selectbox("Y축 속성 선택", options=selected_features, index=1, key="3d_y")
    with col3d_z:
        z_3d = st.selectbox("Z축 속성 선택", options=selected_features, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df,
        x=x_3d,
        y=y_3d,
        z=z_3d,
        color="묶음",
        hover_name="movieNm",
        category_orders={"묶음": ["㉮", "㉯", "㉰"]},
        color_discrete_map={"㉮": "#FF4B4B", "㉯": "#00C0F2", "㉰": "#00D4B1"},
        title=f"3D 산점도 ({x_3d} x {y_3d} x {z_3d})"
    )
    # 점을 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("선택한 속성이 셋보다 적어 3차원 산점도를 표시할 수 없습니다. 3개 이상의 속성을 선택해 주세요.")

st.markdown("---")

# 8. 묶음별 편수 및 네 속성 평균 (원래 단위)
st.subheader("📋 묶음별 요약 표 (원래 단위)")

summary_df = df.groupby("묶음").agg(
    편수=("movieNm", "count"),
    스크린_수_평균=("first_scrn", "mean"),
    누적_관객_평균=("total_audi", "mean"),
    Top10_일수_평균=("days_in_top10", "mean"),
    롱런_지수_평균=("롱런 지수", "mean")
).reindex(["㉮", "㉯", "㉰"])

# 보기 쉽게 포맷팅
formatted_summary = summary_df.copy()
formatted_summary["편수"] = formatted_summary["편수"].map("{:,}편".format)
formatted_summary["스크린_수_평균"] = formatted_summary["스크린_수_평균"].map("{:,.1f}개".format)
formatted_summary["누적_관객_평균"] = formatted_summary["누적_관객_평균"].map("{:,.0f}명".format)
formatted_summary["Top10_일수_평균"] = formatted_summary["Top10_일수_평균"].map("{:.1f}일".format)
formatted_summary["롱런_지수_평균"] = formatted_summary["롱런_지수_평균"].map("{:.2f}".format)

summary_display = formatted_summary.rename(columns={
    "스크린_수_평균": "스크린 수 평균",
    "누적_관객_평균": "누적 관객 평균",
    "Top10_일수_평균": "10위권 일수 평균",
    "롱런_지수_평균": "롱런 지수 평균"
})

st.table(summary_display)

# 9. 묶음별 누적 관객수 상위 5개 영화 제목
st.subheader("🏆 묶음별 누적 관객 상위 5개 영화")

cols = st.columns(3)
for idx, symbol in enumerate(["㉮", "㉯", "㉰"]):
    with cols[idx]:
        st.markdown(f"### 묶음 **{symbol}**")
        top5_movies = df[df["묶음"] == symbol].sort_values(by="total_audi", ascending=False).head(5)
        
        for rank, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"**{rank}. {row['movieNm']}** ({int(row['total_audi']):,}명)")
