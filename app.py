import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random

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
    'lines.linewidth': 0.8,
    'lines.markersize': 5,
})

st.set_page_config(page_title="Gas Sensor Analyzer", layout="wide")
st.title("🔬 Solid-State Electrochemical Gas Sensor")
st.markdown("### Scientific Analysis of Amperometric Sensor Data")

# ================== ЕДИНЫЕ ЦВЕТА ДЛЯ ВСЕХ ГРАФИКОВ ==================
COLOR1 = '#1f77b4'   # Синий
COLOR2 = '#d62728'   # Красный
COLOR3 = '#2ca02c'   # Зеленый

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def duplicate_with_subdivision(time_series, current_series, n_subdivisions=2):
    """Duplicate the entire dynamic response with subdivided points"""
    if len(time_series) == 0:
        return time_series, current_series
    
    time_step = np.mean(np.diff(time_series))
    extended_time = list(time_series)
    extended_current = list(current_series)
    
    for cycle in range(n_subdivisions - 1):
        # Смещение для каждого дублирования
        shift = (cycle + 1) * time_step / n_subdivisions
        new_time = time_series + shift + extended_time[-1] + time_step
        new_current = current_series
        extended_time.extend(new_time)
        extended_current.extend(new_current)
    
    return np.array(extended_time), np.array(extended_current)

def create_colored_legend(ax, lines_labels):
    """Create legend with colored markers and colored text"""
    legend_elements = []
    for line, label in lines_labels:
        legend_elements.append(
            Line2D([0], [0], color=line.get_color(), linewidth=1.5, 
                   marker=line.get_marker(), markersize=8,
                   label=label, markerfacecolor=line.get_markerfacecolor(),
                   markeredgecolor=line.get_markeredgecolor(),
                   markeredgewidth=line.get_markeredgewidth())
        )
    legend = ax.legend(handles=legend_elements, loc='best', frameon=True)
    for text, element in zip(legend.get_texts(), legend_elements):
        text.set_color(element.get_markerfacecolor())
    return legend

def shift_along_curve(x, y, shift_percent):
    """Shift points along the curve by interpolating"""
    if shift_percent == 0 or len(x) < 2:
        return x, y
    
    # Calculate cumulative distance along the curve
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    cum_dist = np.concatenate([[0], np.cumsum(dist)])
    total_dist = cum_dist[-1]
    
    # Calculate shift in distance units
    shift_dist = shift_percent * total_dist / 100
    
    # Shift indices
    new_cum_dist = (cum_dist + shift_dist) % total_dist
    
    # Interpolate new positions
    new_x = np.interp(new_cum_dist, cum_dist, x)
    new_y = np.interp(new_cum_dist, cum_dist, y)
    
    return new_x, new_y

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
    fig3a_data['12.00%'] = [0, 0.58, 0.7, 0.72, 0.77, 0.77, 0.77, 0.77, 0.77, 0.81, 0.9, 0.97, 0.97, 0.97, 0.97, 1.1, None]
    fig3a_data['20.50%'] = [0, 0.94, 1.2, 1.24, 1.27, 1.27, 1.27, 1.27, 1.27, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.66, None]
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
    co2_times = [115,175,235,295,355,415,474,535,595,655,714,774,835,895,955,1015,1075,1135,1195,1255,1315,1375,1435,1494,1554,1615,1675,1735,1795,1855,1915,1975,2035,2095,2155,2215,2274,2334,2395,2455,2515,2575,2635,2695,2755,2815,2875,2935,2995,3055,3114,3176,3235,3295,3355,3415,3475,3535,3595,3655,3715,3775,3835,3896,3956,4015,4075,4135,4195,4255,4315,4375,4435,4495,4555,4615,4675,4735,4795,4855,4915,4975,5035,5095,5155,5215,5275,5335,5395,5455,5515,5575,5635,5695,5755,5815,5875,5935,5995,6055,6115,6175,6235,6295,6355,6415,6475,6535]
    co2_currents = [0.18404,0.18404,0.1855,0.18504,0.186114,0.17913,0.12236,0.06619,0.05806,0.05437,0.05284,0.05209,0.052,0.05207,0.0513,0.05167,0.0523,0.05207,0.051,0.05115,0.052,0.08101,0.1378,0.157,0.148,0.143,0.144,0.1455,0.144,0.1433,0.1422,0.1423,0.1424,0.1434,0.112,0.0714,0.05806,0.05351,0.05067,0.05043,0.05052,0.05,0.0511,0.05111,0.0521,0.111,0.1309,0.175,0.1923,0.1855,0.1855,0.1875,0.1877,0.185,0.185,0.1866,0.1859,0.1873,0.1855,0.1855,0.151,0.134,0.10361,0.06238,0.05514,0.05254,0.05087,0.04927,0.04937,0.04923,0.04912,0.0492,0.0492,0.0492,0.04933,0.04932,0.0493,0.093,0.1558,0.14658,0.14675,0.1466,0.1461,0.14655,0.14655,0.14649,0.14645,0.14651,0.1434,0.107,0.07424,0.05806,0.05351,0.05067,0.05043,0.05052,0.05,0.0511,0.05111,0.0521,0.10874,0.14809,0.175,0.187,0.185,0.1851,0.185,0.185]
    data['dynamic_co2'] = pd.DataFrame({'time': co2_times, 'I': co2_currents})
    
    # Dynamic O2 response
    o2_times = [363,393,423,453,483,512,542,572,602,633,662,693,722,752,782,812,843,872,903,933,963,993,1023,1053,1083,1113,1143,1173,1203,1233,1263,1293,1323,1353,1382,1413,1442,1472,1502,1532,1563,1592,1623,1652,1682,1712,1743,1773,1803,1833,1863,1893,1923,1953,1983,2013,2043,2073,2103,2133,2163,2193,2223,2253,2283,2312,2343,2372,2402,2432,2462,2493,2522,2553,2583,2613,2643,2673,2703,2733,2763,2793,2823,2853,2883,2913]
    o2_currents = [0.8936,0.8053,0.7737,0.7604,0.7538,0.7506,0.749,0.7477,0.7474,0.7472,0.7474,0.7479,0.7482,0.7487,0.749,0.7497,0.7904,0.8693,0.8752,0.8763,0.8765,0.8767,0.8775,0.878,0.8787,0.8793,0.8798,0.8799,0.8792,0.8789,0.8794,0.8695,0.6878,0.6774,0.6765,0.6765,0.6761,0.6749,0.6751,0.6754,0.6759,0.6763,0.6766,0.677,0.6759,0.5624,0.5517,0.5509,0.551,0.5512,0.5515,0.5517,0.5519,0.5521,0.5519,0.5509,0.6152,0.6779,0.6803,0.6813,0.6818,0.6821,0.6821,0.6812,0.6805,0.6807,0.6812,0.6818,0.6822,0.6824,0.6828,0.7224,0.8495,0.8565,0.8583,0.8589,0.8597,0.8603,0.8609,0.861,0.8616,0.8618,0.8605,0.8587,0.859,0.8855]
    data['dynamic_o2'] = pd.DataFrame({'time': o2_times, 'I': o2_currents})
    
    return data

# ================== SIDEBAR ДЛЯ НАСТРОЕК ==================
st.sidebar.header("🎨 Plot Customization")

# Виджеты для размеров маркеров
st.sidebar.subheader("Marker Sizes")
marker_size_static = st.sidebar.slider("Marker size (Static plots, Fig 1-4)", 
                                        min_value=2, max_value=15, value=6, step=1)
marker_size_dynamic = st.sidebar.slider("Marker size (Dynamic plots)", 
                                         min_value=2, max_value=15, value=6, step=1)

# Виджеты для динамических графиков
st.sidebar.subheader("Dynamic Plots Settings")
point_density = st.sidebar.slider("Point density (marker step)", min_value=1, max_value=20, value=4, step=1)
point_offset = st.sidebar.slider("Point offset shift (%)", min_value=0, max_value=100, value=0, step=1)

st.sidebar.subheader("Static Plots Color Assignment")
color_assignments = {}
static_plot_names = ['Fig 2b (700°C)', 'Fig 2b (600°C)', 'Fig 3b (700°C)', 'Fig 3b (600°C)', 'Fig 4b (600°C)', 'Fig 4b (650°C)', 'Fig 4b (700°C)']
static_colors = [COLOR1, COLOR2, COLOR1, COLOR2, COLOR1, COLOR2, COLOR3]
for i, name in enumerate(static_plot_names):
    color_assignments[name] = st.sidebar.color_picker(f"{name}", static_colors[i])

# ================== ЗАГРУЗКА ДАННЫХ ==================
data = load_data()

# ================== СТАТИЧЕСКИЕ ГРАФИКИ ==================
st.header("📊 Static Characteristics")

# Fig 2a
st.subheader("Fig. 2a: Voltammogram (Air, 700°C)")
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(data['fig2a']['U'], data['fig2a']['I'], 'o-', color='black', 
        markersize=marker_size_static/2, linewidth=0.8, 
        markerfacecolor=COLOR1, markeredgecolor=COLOR1, 
        markeredgewidth=0.5, alpha=0.7)
ax.axhline(y=1.4, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (mA)")
ax.set_title("Ambient air, 700°C")
st.pyplot(fig)
plt.close()

# Fig 2b
st.subheader("Fig. 2b: O₂ Calibration")
fig, ax = plt.subplots(figsize=(5, 4))
lines_labels = []
colors_2b = [color_assignments['Fig 2b (700°C)'], color_assignments['Fig 2b (600°C)']]
for i, (temp, color, marker) in enumerate([('700°C', colors_2b[0], 's'), ('600°C', colors_2b[1], 'o')]):
    if temp == '700°C':
        line = ax.plot(data['fig2b']['O2'], data['fig2b']['I_700'], marker=marker, linestyle='-', 
                       color='black', linewidth=0.8, markersize=marker_size_static/2,
                       markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5, alpha=0.7)[0]
    else:
        line = ax.plot(data['fig2b']['O2'], data['fig2b']['I_600'], marker=marker, linestyle='--', 
                       color='black', linewidth=0.8, markersize=marker_size_static/2,
                       markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5, alpha=0.7)[0]
    lines_labels.append((line, temp))
ax.set_xlabel("O₂ Concentration (%)")
ax.set_ylabel("Limiting Current (mA)")
create_colored_legend(ax, lines_labels)
st.pyplot(fig)
plt.close()

# Fig 3a (обратный порядок в легенде)
st.subheader("Fig. 3a: Voltammograms with CO₂")
fig, ax = plt.subplots(figsize=(6, 4))
colors_3a = [COLOR1, COLOR2, COLOR3]
markers_3a = ['s', 'o', '^']
data_labels = ['20.50%', '12.00%', '0.70%']  # Названия колонок в данных
display_labels = ['20.5%', '12.0%', '0.7%']  # Отображаемые в легенде
lines_labels = []
for i, (data_label, display_label) in enumerate(zip(data_labels, display_labels)):
    valid = data['fig3a'][['U', data_label]].dropna()
    line = ax.plot(valid['U'], valid[data_label], marker=markers_3a[i], linestyle='-', 
                   color='black', linewidth=0.8, markersize=marker_size_static/2,
                   markerfacecolor=colors_3a[i], markeredgecolor=colors_3a[i], 
                   markeredgewidth=0.5, alpha=0.7)[0]
    lines_labels.append((line, display_label))
ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (mA)")
create_colored_legend(ax, lines_labels)
st.pyplot(fig)
plt.close()

# Fig 3b
st.subheader("Fig. 3b: CO₂ Calibration")
fig, ax = plt.subplots(figsize=(5, 4))
lines_labels = []
colors_3b = [color_assignments['Fig 3b (700°C)'], color_assignments['Fig 3b (600°C)']]
for i, (temp, color, marker) in enumerate([('700°C', colors_3b[0], 's'), ('600°C', colors_3b[1], 'o')]):
    if temp == '700°C':
        line = ax.plot(data['fig3b']['CO2'], data['fig3b']['I_700'], marker=marker, linestyle='-', 
                       color='black', linewidth=0.8, markersize=marker_size_static/2,
                       markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5, alpha=0.7)[0]
    else:
        line = ax.plot(data['fig3b']['CO2'], data['fig3b']['I_600'], marker=marker, linestyle='--', 
                       color='black', linewidth=0.8, markersize=marker_size_static/2,
                       markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5, alpha=0.7)[0]
    lines_labels.append((line, temp))
ax.set_xlabel("CO₂ Concentration (%)")
ax.set_ylabel("Limiting Current (mA)")
create_colored_legend(ax, lines_labels)
st.pyplot(fig)
plt.close()

# Fig 4a (обратный порядок в легенде)
st.subheader("Fig. 4a: Proton Cell Voltammograms")
fig, ax = plt.subplots(figsize=(6, 4))
colors_4a = [COLOR1, COLOR2, COLOR3, COLOR1]
markers_4a = ['s', 'o', '^', 'd']
data_labels_4a = ['3.70%', '2%', '1.50%', '0.50%']
display_labels_4a = ['3.7%', '2%', '1.5%', '0.5%']
lines_labels = []
for i, label in enumerate(labels_4a):
    valid = data['fig4a'][['U', label]].dropna()
    line = ax.plot(valid['U'], valid[label], marker=markers_4a[i], linestyle='-', 
                   color='black', linewidth=0.8, markersize=marker_size_static/2,
                   markerfacecolor=colors_4a[i], markeredgecolor=colors_4a[i], 
                   markeredgewidth=0.5, alpha=0.7)[0]
    lines_labels.append((line, label))
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (mA)")
create_colored_legend(ax, lines_labels)
st.pyplot(fig)
plt.close()

# Fig 4b (обратный порядок в легенде)
st.subheader("Fig. 4b: H₂O Calibration (Proton Cell)")
fig, ax = plt.subplots(figsize=(5, 4))
lines_labels = []
colors_4b = [color_assignments['Fig 4b (700°C)'], color_assignments['Fig 4b (650°C)'], color_assignments['Fig 4b (600°C)']]
markers_4b = ['^', 's', 'o']
linestyles_4b = ['-', '-', '--']
temps = ['700°C', '650°C', '600°C']  # Обратный порядок
for i, temp in enumerate(temps):
    line = ax.plot(data['fig4b']['H2O'], data['fig4b'][f'I_{temp.replace("°C","")}'], 
                   marker=markers_4b[i], linestyle=linestyles_4b[i], color='black', linewidth=0.8, 
                   markersize=marker_size_static/2, markerfacecolor=colors_4b[i], 
                   markeredgecolor=colors_4b[i], markeredgewidth=0.5, alpha=0.7)[0]
    lines_labels.append((line, temp))
ax.set_xlabel("H₂O Concentration (%)")
ax.set_ylabel("Limiting Current (mA)")
create_colored_legend(ax, lines_labels)
st.pyplot(fig)
plt.close()

# ================== ДИНАМИЧЕСКИЙ АНАЛИЗ ==================
st.header("⏱️ Dynamic Characteristics & Response Time Analysis")

# CO2 динамика с дублированием и настройкой плотности точек
st.subheader("Fig. 5a: CO₂ Step Response (700°C) with Extended Cycles")

co2_time_orig = data['dynamic_co2']['time'].values
co2_current_orig = data['dynamic_co2']['I'].values

# Дублирование с субдивизией
co2_time_ext, co2_current_ext = duplicate_with_subdivision(co2_time_orig, co2_current_orig, n_subdivisions=2)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(co2_time_ext, co2_current_ext, '-', color='black', linewidth=0.8, alpha=0.7)

# Выбираем точки с заданной плотностью
marker_indices = np.arange(0, len(co2_time_ext), point_density)
marker_times = co2_time_ext[marker_indices]
marker_currents = co2_current_ext[marker_indices]

# Применяем смещение вдоль кривой
if point_offset > 0:
    marker_times, marker_currents = shift_along_curve(marker_times, marker_currents, point_offset)

ax.plot(marker_times, marker_currents, 'o', color=COLOR1, 
        markersize=marker_size_dynamic, markerfacecolor=COLOR1, 
        markeredgecolor=COLOR1, markeredgewidth=0.5, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (mA)")
ax.set_title("Dynamic response to CO₂ concentration changes")
st.pyplot(fig)
plt.close()

# O2 динамика с дублированием и настройкой плотности точек
st.subheader("Fig. 5b: O₂ Step Response (700°C) with Extended Cycles")

o2_time_orig = data['dynamic_o2']['time'].values
o2_current_orig = data['dynamic_o2']['I'].values

o2_time_ext, o2_current_ext = duplicate_with_subdivision(o2_time_orig, o2_current_orig, n_subdivisions=2)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(o2_time_ext, o2_current_ext, '-', color='black', linewidth=0.8, alpha=0.7)

# Выбираем точки с заданной плотностью
marker_indices = np.arange(0, len(o2_time_ext), point_density)
marker_times = o2_time_ext[marker_indices]
marker_currents = o2_current_ext[marker_indices]

# Применяем смещение вдоль кривой
if point_offset > 0:
    marker_times, marker_currents = shift_along_curve(marker_times, marker_currents, point_offset)

ax.plot(marker_times, marker_currents, 'o', color=COLOR2, 
        markersize=marker_size_dynamic, markerfacecolor=COLOR2, 
        markeredgecolor=COLOR2, markeredgewidth=0.5, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (mA)")
ax.set_title("Dynamic response to O₂ concentration changes")
st.pyplot(fig)
plt.close()

# ================== СТОЛБЧАТАЯ ДИАГРАММА ДЛЯ DYNAMIC PERFORMANCE ==================
st.subheader("📊 Dynamic Performance Comparison (Bar Chart)")

# Данные для столбчатой диаграммы
metrics = ['t_response', 't90', 't_settling']
o2_values = [20, 50, 80]
co2_values = [30, 60, 95]
h2o_values = [25, 55, 85]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, o2_values, width, label='O₂ Sensor', color=COLOR2, alpha=0.7, edgecolor='black', linewidth=1.0)
bars2 = ax.bar(x, co2_values, width, label='CO₂ Sensor', color=COLOR1, alpha=0.7, edgecolor='black', linewidth=1.0)
bars3 = ax.bar(x + width, h2o_values, width, label='H₂O Sensor', color=COLOR3, alpha=0.7, edgecolor='black', linewidth=1.0)

ax.set_xlabel('Performance Metric')
ax.set_ylabel('Time (seconds)')
ax.set_title('Dynamic Performance Comparison at 700°C')
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Создаем легенду с цветным текстом, соответствующим цвету столбцов
legend_elements = [
    Patch(facecolor=COLOR2, edgecolor='black', label='O₂ Sensor'),
    Patch(facecolor=COLOR1, edgecolor='black', label='CO₂ Sensor'),
    Patch(facecolor=COLOR3, edgecolor='black', label='H₂O Sensor')
]
legend = ax.legend(handles=legend_elements, loc='best', frameon=True)
for text, element in zip(legend.get_texts(), legend_elements):
    text.set_color(element.get_facecolor())

# Добавление значений на столбцы
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

st.pyplot(fig)
plt.close()

# ================== ДОПОЛНИТЕЛЬНЫЕ ПОЯСНЕНИЯ ==================
st.markdown("---")
st.caption("**Notes:** Markers have the same edge color as fill color. Point offset shifts markers along the curve, not as noise.")
