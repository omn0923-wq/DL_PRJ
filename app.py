"""따릉이 수요 예측 시뮬레이터 (Streamlit)
노트북에서 학습한 LSTM 가중치(model_weights.npz)를 numpy로 불러 추론한다. (TensorFlow 불필요)
실행 : streamlit run app.py
"""

import json
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="따릉이 수요 예측 시뮬레이터", page_icon="🚲", layout="wide"
)
GREEN, ORANGE, GRAY = "#1E4D3B", "#E8743B", "#9CA3AF"
KR = holidays.country_holidays("KR", years=range(2017, 2029))


# ---------- 데이터·모델 불러오기 ----------
@st.cache_resource
def load():
    meta = json.load(open("meta.json", encoding="utf-8"))
    w = np.load("model_weights.npz")
    daily = pd.read_csv("daily_dataset.csv", parse_dates=["date"]).set_index("date")
    test = pd.read_csv("test_pred.csv", parse_dates=["date"])
    return meta, {k: w[k] for k in w.files}, daily, test


meta, P, daily, test = load()
W, FEATURES = meta["window"], meta["features"]
x_mean, x_std = np.array(meta["x_mean"]), np.array(meta["x_std"])
y_mean, y_std = meta["y_mean"], meta["y_std"]


# ---------- 모델 (노트북과 동일) ----------
def make_features(raw):
    f = pd.DataFrame(index=raw.index)
    for c in ["temp_mean", "temp_max", "temp_min", "wind", "humid"]:
        f[c] = raw[c]
    f["rain_log"] = np.log1p(raw["rain"])
    f["offday"] = [int(d.weekday() >= 5 or d.date() in KR) for d in raw.index]
    f["dow_sin"] = np.sin(2 * np.pi * raw.index.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * raw.index.dayofweek / 7)
    f["month_sin"] = np.sin(2 * np.pi * (raw.index.month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (raw.index.month - 1) / 12)
    return f[FEATURES]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_np(x, k, r, b, return_seq):
    B, T, _ = x.shape
    u = r.shape[0]
    hh, cc, outs = np.zeros((B, u)), np.zeros((B, u)), []
    for t in range(T):
        zz = x[:, t] @ k + hh @ r + b
        i, f, g, o = np.split(zz, 4, axis=1)
        cc = sigmoid(f) * cc + sigmoid(i) * np.tanh(g)
        hh = sigmoid(o) * np.tanh(cc)
        outs.append(hh)
    return np.stack(outs, 1) if return_seq else hh


def forward(seq, e):
    h1 = lstm_np(seq, P["l1_k"], P["l1_r"], P["l1_b"], True)
    h2 = lstm_np(h1, P["l2_k"], P["l2_r"], P["l2_b"], False)
    z = np.concatenate([h2, np.maximum(e @ P["e_k"] + P["e_b"], 0)], 1)
    z = np.maximum(z @ P["d_k"] + P["d_b"], 0)
    return (z @ P["o_k"] + P["o_b"])[:, 0]


def history_seq(target):
    """대상일 직전 W일의 [대여량, 기상·달력] 시퀀스"""
    hist = daily.loc[target - timedelta(days=W) : target - timedelta(days=1)]
    f = (make_features(hist) - x_mean) / x_std
    rides_s = ((hist["rides"] - y_mean) / y_std).values
    return np.column_stack([rides_s, f.values])[None]


def predict(target, wx):
    """wx : 대상일 날씨 dict(temp_mean, temp_max, temp_min, rain, wind, humid)"""
    raw = pd.DataFrame([wx], index=pd.DatetimeIndex([target]))
    e = ((make_features(raw) - x_mean) / x_std).values
    return float(forward(history_seq(target), e)[0] * y_std + y_mean)


def shifted(wx, **kw):
    """평균기온을 바꾸면 최고·최저기온도 같은 폭만큼 이동"""
    w = dict(wx)
    if "temp_mean" in kw:
        d = kw["temp_mean"] - w["temp_mean"]
        w["temp_max"] += d
        w["temp_min"] += d
    w.update(kw)
    return w


# ---------- 화면 ----------
st.title("🚲 날씨 기반 따릉이 수요 예측 시뮬레이터")
st.caption(
    f'서울 공공자전거 일별 대여건수 · 학습 데이터 {meta["data_start"]} ~ {meta["data_end"]} · LSTM 예측모델'
)

first = daily.index.min() + timedelta(days=W)
last = daily.index.max()
with st.sidebar:
    st.header("시나리오 입력")
    target = pd.Timestamp(
        st.date_input(
            "예측 대상일",
            value=last.date(),
            min_value=first.date(),
            max_value=(last + timedelta(days=1)).date(),
            help="데이터 마지막 날의 다음 날까지 예측할 수 있습니다.",
        )
    )
    in_data = target in daily.index
    ref = daily.loc[target] if in_data else daily.iloc[-1]  # 기본값으로 쓸 날씨
    st.caption(
        "학습 데이터의 평균기온은 -12~33℃ 범위입니다. 원래 날씨(실측)가 기본값이고, 값을 바꾸면 시나리오가 됩니다."
        if in_data
        else "데이터 마지막 날의 날씨를 기본값으로 사용합니다."
    )
    k = target.strftime("%Y%m%d")
    t_mean = st.slider(
        "평균기온 (℃)", -15.0, 35.0, float(ref["temp_mean"]), 0.5, key=f"t{k}"
    )
    rain = st.slider("일강수량 (mm)", 0.0, 150.0, float(ref["rain"]), 0.5, key=f"r{k}")
    wind = st.slider("평균풍속 (m/s)", 0.0, 15.0, float(ref["wind"]), 0.1, key=f"w{k}")
    humid = st.slider("평균습도 (%)", 0.0, 100.0, float(ref["humid"]), 1.0, key=f"h{k}")

base_wx = {
    c: float(ref[c])
    for c in ["temp_mean", "temp_max", "temp_min", "rain", "wind", "humid"]
}
scn_wx = shifted(base_wx, temp_mean=t_mean, rain=rain, wind=wind, humid=humid)
p_base, p_scn = predict(target, base_wx), predict(target, scn_wx)
holiday_name = KR.get(target.date())
day_type = (
    f"공휴일({holiday_name})"
    if holiday_name
    else ("주말" if target.weekday() >= 5 else "평일")
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["시나리오 시뮬레이터", "기온·강수 민감도", "모델 성능", "데이터"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "시나리오 예측 대여건수",
        f"{p_scn:,.0f}건",
        f"{(p_scn / p_base - 1) * 100:+.1f}% (원래 날씨 대비)",
    )
    c2.metric("원래 날씨 예측", f"{p_base:,.0f}건")
    c3.metric(
        "실제 대여건수",
        f'{daily.loc[target, "rides"]:,.0f}건' if in_data else "아직 없음",
    )
    c4.metric("대상일", target.strftime("%Y-%m-%d (%a)"), day_type, delta_color="off")

    hist = daily.loc[target - timedelta(days=W) : target - timedelta(days=1), "rides"]
    fig = go.Figure()
    fig.add_scatter(
        x=hist.index,
        y=hist.values,
        mode="lines+markers",
        name=f"최근 {W}일 실제",
        line=dict(color=GRAY),
    )
    if in_data:
        fig.add_scatter(
            x=[target],
            y=[daily.loc[target, "rides"]],
            mode="markers",
            name="대상일 실제",
            marker=dict(color=GREEN, size=12),
        )
    fig.add_scatter(
        x=[target],
        y=[p_base],
        mode="markers",
        name="원래 날씨 예측",
        marker=dict(color=GREEN, size=12, symbol="diamond-open", line=dict(width=2)),
    )
    fig.add_scatter(
        x=[target],
        y=[p_scn],
        mode="markers",
        name="시나리오 예측",
        marker=dict(color=ORANGE, size=14, symbol="diamond"),
    )
    fig.update_layout(
        height=420,
        margin=dict(t=30, b=10),
        yaxis_title="일 대여건수",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.write(
        f"**{target:%Y-%m-%d}** 의 다른 조건은 사이드바 값으로 고정하고, 한 가지 변수만 바꿨을 때의 예측 대여건수입니다."
    )
    g_t, g_r = np.arange(-10, 35, 2.0), np.array(
        [0, 1, 2, 5, 10, 15, 20, 30, 40, 60, 80]
    )
    y_t = [predict(target, shifted(scn_wx, temp_mean=v)) for v in g_t]
    y_r = [predict(target, shifted(scn_wx, rain=float(v))) for v in g_r]
    a, b = st.columns(2)
    f1 = go.Figure(
        go.Scatter(x=g_t, y=y_t, mode="lines+markers", line=dict(color=GREEN))
    )
    f1.add_vline(x=t_mean, line_dash="dash", line_color=ORANGE)
    f1.update_layout(
        title="평균기온에 따른 예측",
        xaxis_title="평균기온 (℃)",
        yaxis_title="예측 대여건수",
        height=380,
        margin=dict(t=50),
    )
    f2 = go.Figure(
        go.Scatter(x=g_r, y=y_r, mode="lines+markers", line=dict(color=ORANGE))
    )
    f2.add_vline(x=rain, line_dash="dash", line_color=GREEN)
    f2.update_layout(
        title="강수량에 따른 예측",
        xaxis_title="일강수량 (mm)",
        yaxis_title="예측 대여건수",
        height=380,
        margin=dict(t=50),
    )
    a.plotly_chart(f1, width="stretch")
    b.plotly_chart(f2, width="stretch")
    tbl = pd.DataFrame(
        {
            "강수량(mm)": g_r,
            "예측 대여건수": np.round(y_r, 0),
            "맑은 날 대비(%)": np.round((np.array(y_r) / y_r[0] - 1) * 100, 1),
        }
    )
    st.dataframe(tbl, hide_index=True, width="stretch")

with tab3:
    m = pd.DataFrame(meta["metrics"]).T
    st.write(
        f'테스트 구간 {meta["test_start"]} ~ {meta["test_end"]} (학습에 사용하지 않은 마지막 15%)'
    )
    st.dataframe(
        m.style.format(
            {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "MAPE(%)": "{:.1f}"}
        ),
        width="stretch",
    )
    f3 = go.Figure()
    f3.add_scatter(
        x=test["date"], y=test["actual"], name="실제", line=dict(color=GREEN)
    )
    f3.add_scatter(
        x=test["date"], y=test["lstm"], name="LSTM 예측", line=dict(color=ORANGE)
    )
    f3.update_layout(
        height=400,
        margin=dict(t=30),
        yaxis_title="일 대여건수",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(f3, width="stretch")

with tab4:
    st.write("일별 대여건수와 서울 기상 데이터 (병합본)")
    st.dataframe(
        daily.round(1).sort_index(ascending=False), width="stretch", height=420
    )
    st.caption(
        "출처 : 서울 열린데이터광장 공공자전거 일별 대여건수, 기상청 기상자료개방포털 ASOS 일자료(서울)"
    )
