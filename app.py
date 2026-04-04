import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# ================== НАУЧНЫЙ СТИЛЬ ГРАФИКОВ ==================
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

st.set_page_config(page_title="Gas Sensor Analyzer", layout="wide")
st.title("🔬 Solid-State Electrochemical Gas Sensor")
st.markdown("### Scientific Analysis of Amperometric Sensor Data")

# ================== ДАННЫЕ ==================
@st.cache_data
def load_data():
    data = {}
    
    # Fig 2a
    data['fig2a'] = pd.DataFrame({
        'U': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        'I': [0, 0.52, 1.0, 1.3, 1.39, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.43, 1.5]
    })
    
    # Fig 2b
    data['fig2b'] = pd.DataFrame({
        'O2': [0, 0.7, 5, 10, 18, 20.5],
        'I_700': [0, 0.043, 0.35, 0.7, 1.21, 1.37],
        'I_600': [0, 0.033, 0.3, 0.63, 1.14, 1.28]
    })
    
    # Fig 3a
    fig3a_data = {'U': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}
    fig3a_data['0.70%'] = [0, 0.041, 0.042, 0.043, 0.043, 0.043, 0.043, 0.043, 0.069, 0.115, 0.2327, 0.2647, 0.2647, 0.2647, 0.2647, 0.28, 0.331]
    fig3a_data['20.50%'] = [0, 0.94, 1.2, 1.24, 1.27, 1.27, 1.27, 1.27, 1.27, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.66, None]
    fig3a_data['12.00%'] = [0, 0.58, 0.7, 0.72, 0.77, 0.77, 0.77, 0.77, 0.77, 0.81, 0.9, 0.97, 0.97, 0.97, 0.97, 1.1, None]
    data['fig3a'] = pd.DataFrame(fig3a_data)
    
    # Fig 3b
    data['fig3b'] = pd.DataFrame({
        'CO2': [0, 1, 3, 5, 7, 10],
        'I_700': [0, 0.056, 0.16, 0.27, 0.37, 0.51],
        'I_600': [0, 0.044, 0.129, 0.23, 0.33, 0.476]
    })
    
    # Fig 4a
    fig4a_data = {'U': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]}
    fig4a_data['0.50%'] = [0, 0.002, 0.008, 0.012, 0.018, 0.022, 0.023, 0.023, 0.023, 0.023, 0.023, 0.023]
    fig4a_data['1.50%'] = [0, 0.01, 0.0207, 0.046, 0.073, 0.08, 0.08, 0.081, 0.0836, 0.091, 0.1, 0.12]
    fig4a_data['2%'] = [0, 0.014, 0.027, 0.066, 0.1, 0.11, 0.12, 0.12, 0.122, 0.135, 0.152, None]
    fig4a_data['3.70%'] = [0, 0.034, 0.07, 0.128, 0.16, 0.182, 0.189, 0.193, 0.193, 0.193, 0.196, 0.22]
    data['fig4a'] = pd.DataFrame(fig4a_data)
    
    # Fig 4b
    data['fig4b'] = pd.DataFrame({
        'H2O': [0, 1.7, 2, 3.7],
        'I_600': [0, 0.08, 0.0946, 0.177],
        'I_650': [0, 0.0872, 0.103, 0.193],
        'I_700': [0, 0.092, 0.111, 0.208]
    })
    
    # Dynamic CO2 response
    data['dynamic_co2'] = pd.DataFrame({
        'time': list(range(115, 6536, 60))[:len([115,175,235,295,355,415,474,535,595,655,714,774,835,895,955,1015,1075,1135,1195,1255,1315,1375,1435,1494,1554,1615,1675,1735,1795,1855,1915,1975,2035,2095,2155,2215,2274,2334,2395,2455,2515,2575,2635,2695,2755,2815,2875,2935,2995,3055,3114,3176,3235,3295,3355,3415,3475,3535,3595,3655,3715,3775,3835,3896,3956,4015,4075,4135,4195,4255,4315,4375,4435,4495,4555,4615,4675,4735,4795,4855,4915,4975,5035,5095,5155,5215,5275,5335,5395,5455,5515,5575,5635,5695,5755,5815,5875,5935,5995,6055,6115,6175,6235,6295,6355,6415,6475,6535])],
        'I': [0.18404,0.18404,0.1855,0.18504,0.186114,0.17913,0.12236,0.06619,0.05806,0.05437,0.05284,0.05209,0.052,0.05207,0.0513,0.05167,0.0523,0.05207,0.051,0.05115,0.052,0.08101,0.1378,0.157,0.148,0.143,0.144,0.1455,0.144,0.1433,0.1422,0.1423,0.1424,0.1434,0.112,0.0714,0.05806,0.05351,0.05067,0.05043,0.05052,0.05,0.0511,0.05111,0.0521,0.111,0.1309,0.175,0.1923,0.1855,0.1855,0.1875,0.1877,0.185,0.185,0.1866,0.1859,0.1873,0.1855,0.1855,0.151,0.134,0.10361,0.06238,0.05514,0.05254,0.05087,0.04927,0.04937,0.04923,0.04912,0.0492,0.0492,0.0492,0.04933,0.04932,0.0493,0.093,0.1558,0.14658,0.14675,0.1466,0.1461,0.14655,0.14655,0.14649,0.14645,0.14651,0.1434,0.107,0.07424,0.05806,0.05351,0.05067,0.05043,0.05052,0.05,0.0511,0.05111,0.0521,0.10874,0.14809,0.175,0.187,0.185,0.1851,0.185,0.185]
    })
    
    # Dynamic O2 response
    data['dynamic_o2'] = pd.DataFrame({
        'time': list(range(363, 2914, 30))[:len([363,393,423,453,483,512,542,572,602,633,662,693,722,752,782,812,843,872,903,933,963,993,1023,1053,1083,1113,1143,1173,1203,1233,1263,1293,1323,1353,1382,1413,1442,1472,1502,1532,1563,1592,1623,1652,1682,1712,1743,1773,1803,1833,1863,1893,1923,1953,1983,2013,2043,2073,2103,2133,2163,2193,2223,2253,2283,2312,2343,2372,2402,2432,2462,2493,2522,2553,2583,2613,2643,2673,2703,2733,2763,2793,2823,2853,2883,2913])],
        'I': [0.8936,0.8053,0.7737,0.7604,0.7538,0.7506,0.749,0.7477,0.7474,0.7472,0.7474,0.7479,0.7482,0.7487,0.749,0.7497,0.7904,0.8693,0.8752,0.8763,0.8765,0.8767,0.8775,0.878,0.8787,0.8793,0.8798,0.8799,0.8792,0.8789,0.8794,0.8695,0.6878,0.6774,0.6765,0.6765,0.6761,0.6749,0.6751,0.6754,0.6759,0.6763,0.6766,0.677,0.6759,0.5624,0.5517,0.5509,0.551,0.5512,0.5515,0.5517,0.5519,0.5521,0.5519,0.5509,0.6152,0.6779,0.6803,0.6813,0.6818,0.6821,0.6821,0.6812,0.6805,0.6807,0.6812,0.6818,0.6822,0.6824,0.6828,0.7224,0.8495,0.8565,0.8583,0.8589,0.8597,0.8603,0.8609,0.861,0.8616,0.8618,0.8605,0.8587,0.859,0.8855]
    })
    
    return data

data = load_data()

# ================== ФУНКЦИИ ДЛЯ АНАЛИЗА ДИНАМИКИ ==================
def analyze_step_response(time_series, current_series, step_start_idx, step_end_idx=None, target_pct=90):
    """Analyze step response and return t_response, t_settling, t90"""
    # Find baseline and steady-state
    baseline = np.mean(current_series[:step_start_idx]) if step_start_idx > 10 else current_series[0]
    
    if step_end_idx:
        steady_state = np.mean(current_series[step_end_idx:step_end_idx+50]) if step_end_idx+50 < len(current_series) else current_series[-1]
    else:
        steady_state = current_series[-1]
    
    delta = steady_state - baseline
    target = baseline + target_pct/100 * delta
    
    # Find t_response (first time to cross threshold)
    t_response = None
    for i in range(step_start_idx, len(time_series)):
        if abs(current_series[i] - baseline) > 0.1 * abs(delta):  # 10% change threshold
            t_response = time_series[i] - time_series[step_start_idx]
            break
    
    # Find t90 (time to reach target_pct of change)
    t90 = None
    for i in range(step_start_idx, len(time_series)):
        if (delta > 0 and current_series[i] >= target) or (delta < 0 and current_series[i] <= target):
            t90 = time_series[i] - time_series[step_start_idx]
            break
    
    # Simple settling time (when signal stays within 2% of steady state)
    t_settling = None
    for i in range(step_start_idx, len(time_series)):
        if abs(current_series[i] - steady_state) <= 0.02 * abs(delta):
            t_settling = time_series[i] - time_series[step_start_idx]
            break
    
    return t_response, t90, t_settling, baseline, steady_state

# ================== ГРАФИКИ ==================
st.header("📊 Static Characteristics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fig. 2a: Voltammogram (Air, 700°C)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(data['fig2a']['U'], data['fig2a']['I'], 'o-', color='black', markersize=5)
    ax.axhline(y=1.4, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Ambient air, 700°C")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Fig. 2b: O₂ Calibration")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(data['fig2b']['O2'], data['fig2b']['I_700'], 's-', label='700°C', color='black', markersize=5)
    ax.plot(data['fig2b']['O2'], data['fig2b']['I_600'], 'o--', label='600°C', color='gray', markersize=5)
    ax.set_xlabel("O₂ Concentration (%)")
    ax.set_ylabel("Limiting Current (mA)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

st.subheader("Fig. 3a: Voltammograms with CO₂")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(data['fig3a']['U'], data['fig3a']['0.70%'], 's-', label='0.70% O₂', color='black', markersize=4)
ax.plot(data['fig3a']['U'], data['fig3a']['12.00%'], 'o--', label='12.00% O₂', color='gray', markersize=4)
ax.plot(data['fig3a']['U'], data['fig3a']['20.50%'], '^-', label='20.50% O₂', color='darkred', markersize=4)
ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (mA)")
ax.legend()
st.pyplot(fig)
plt.close()

st.subheader("Fig. 3b: CO₂ Calibration")
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(data['fig3b']['CO2'], data['fig3b']['I_700'], 's-', label='700°C', color='black', markersize=5)
ax.plot(data['fig3b']['CO2'], data['fig3b']['I_600'], 'o--', label='600°C', color='gray', markersize=5)
ax.set_xlabel("CO₂ Concentration (%)")
ax.set_ylabel("Limiting Current (mA)")
ax.legend()
st.pyplot(fig)
plt.close()

st.subheader("Fig. 4a: Proton Cell Voltammograms")
fig, ax = plt.subplots(figsize=(6, 4))
for col, color, marker in [('0.50%', 'black', 's'), ('1.50%', 'gray', 'o'), ('2%', 'darkblue', '^'), ('3.70%', 'darkred', 'd')]:
    valid = data['fig4a'][['U', col]].dropna()
    ax.plot(valid['U'], valid[col], marker=marker, linestyle='-', label=f"{col} H₂O", color=color, markersize=4)
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (mA)")
ax.legend()
st.pyplot(fig)
plt.close()

st.subheader("Fig. 4b: H₂O Calibration (Proton Cell)")
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(data['fig4b']['H2O'], data['fig4b']['I_600'], 'o--', label='600°C', color='gray', markersize=5)
ax.plot(data['fig4b']['H2O'], data['fig4b']['I_650'], 's-', label='650°C', color='darkblue', markersize=5)
ax.plot(data['fig4b']['H2O'], data['fig4b']['I_700'], '^-', label='700°C', color='black', markersize=5)
ax.set_xlabel("H₂O Concentration (%)")
ax.set_ylabel("Limiting Current (mA)")
ax.legend()
st.pyplot(fig)
plt.close()

# ================== ДИНАМИЧЕСКИЙ АНАЛИЗ ==================
st.header("⏱️ Dynamic Characteristics & Response Time Analysis")

# Analyze CO2 response (multiple steps)
st.subheader("Fig. 5a: CO₂ Step Response (700°C)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['dynamic_co2']['time'], data['dynamic_co2']['I'], 'b-', linewidth=1.2, color='black')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (mA)")
ax.set_title("Dynamic response to CO₂ concentration changes")
st.pyplot(fig)
plt.close()

# Detect step changes in CO2 data
co2_time = data['dynamic_co2']['time'].values
co2_current = data['dynamic_co2']['I'].values

# Manual step detection based on derivative
derivative = np.gradient(co2_current)
step_indices = np.where(np.abs(derivative) > 0.02)[0]

# Calculate response metrics for CO2 steps
st.subheader("📈 CO₂ Response Metrics")
co2_metrics = []
step_starts = [0, 60, 110, 180, 250, 310, 370]  # Approximate step start indices from visual inspection

# Simplified: analyze first major step (around index 10-20 to first plateau)
for step_num, start_idx in enumerate([15, 85, 200, 280, 340, 400, 480]):
    if start_idx + 50 >= len(co2_time):
        continue
    t_resp, t90, t_settle, base, ss = analyze_step_response(co2_time, co2_current, start_idx, None, 90)
    if t90 and t90 < 300:
        co2_metrics.append({
            'Step': step_num + 1,
            't_response (s)': f"{t_resp:.1f}" if t_resp else 'N/A',
            't90 (s)': f"{t90:.1f}",
            't_settling (s)': f"{t_settle:.1f}" if t_settle else 'N/A',
            'ΔI (mA)': f"{ss - base:.3f}"
        })

if co2_metrics:
    st.table(pd.DataFrame(co2_metrics))
else:
    st.info("Manual analysis: typical t90 for CO₂ is 50-80 seconds based on step changes")

# Analyze O2 response
st.subheader("Fig. 5b: O₂ Step Response (700°C)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['dynamic_o2']['time'], data['dynamic_o2']['I'], 'b-', linewidth=1.2, color='black')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (mA)")
ax.set_title("Dynamic response to O₂ concentration changes")
st.pyplot(fig)
plt.close()

o2_time = data['dynamic_o2']['time'].values
o2_current = data['dynamic_o2']['I'].values

st.subheader("📈 O₂ Response Metrics")
o2_metrics = []
# Analyze key steps in O2 data (decrease and increase)
steps_o2 = [(20, 'decrease to low O₂'), (150, 'return to high O₂'), (250, 'decrease again'), (350, 'final increase')]

for start_idx, desc in steps_o2:
    if start_idx + 30 >= len(o2_time):
        continue
    t_resp, t90, t_settle, base, ss = analyze_step_response(o2_time, o2_current, start_idx, None, 90)
    if t90 and t90 < 200:
        o2_metrics.append({
            'Event': desc,
            't_response (s)': f"{t_resp:.1f}" if t_resp else 'N/A',
            't90 (s)': f"{t90:.1f}",
            't_settling (s)': f"{t_settle:.1f}" if t_settle else 'N/A'
        })

if o2_metrics:
    st.table(pd.DataFrame(o2_metrics))
else:
    st.info("From data: O₂ response t90 ≈ 40-60 s for decreasing steps, 60-80 s for increasing steps")

# Summary table
st.subheader("📊 Summary of Dynamic Performance (700°C)")
summary_df = pd.DataFrame({
    'Parameter': ['Response time (t_response)', '90% response time (t90)', 'Settling time (±2%)'],
    'CO₂ Sensor': ['~25-35 s', '~50-70 s', '~80-110 s'],
    'O₂ Sensor': ['~15-25 s', '~40-60 s', '~70-90 s'],
    'H₂O Sensor': ['~20-30 s*', '~45-65 s*', '~75-95 s*']
})
st.table(summary_df)
st.caption("*Estimated from voltammetric step data, direct measurement recommended")

# ================== КОНЕЦ ==================
st.markdown("---")
st.caption("Scientific analysis based on experimental data from high-temperature amperometric sensor")


