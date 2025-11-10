import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt  # 1. 导入图表库

# --- (新增) 设置 Matplotlib 中文字体 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode.minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置失败 (可能未安装 'SimHei')，图表标签可能显示异常: {e}")
# --- 结束新增 ---

# --- 1. 定义常量 ---
CAP_PV_A = 750.0
CAP_Wind_B = 1000.0
CAP_PV_C = 600.0
CAP_Wind_C = 500.0

PRICE_GRID = 1.0
PRICE_PV = 0.4
PRICE_Wind = 0.5

# 储能投资成本
COST_P_ES = 800.0
COST_E_ES = 1800.0
LIFESPAN_DAYS = 10 * 365

# 附件文件名
FILE_LOAD = "附件1：各园区典型日负荷数据.xlsx"
FILE_GEN = "附件2：各园区典型日风光发电数据.xlsx"

# --- 2. 加载和准备数据 ---
try:
    df_load = pd.read_excel(FILE_LOAD)
    df_gen_pu = pd.read_excel(FILE_GEN, header=None, skiprows=3)
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 准备DataFrame
df = pd.DataFrame()
df['Load_A'] = df_load['园区A负荷(kW)']
df['Load_B'] = df_load['园区B负荷(kW)']
df['Load_C'] = df_load['园区C负荷(kW)']

df['PV_A_pu'] = pd.to_numeric(df_gen_pu[1])
df['Wind_B_pu'] = pd.to_numeric(df_gen_pu[2])
df['PV_C_pu'] = pd.to_numeric(df_gen_pu[3])
df['Wind_C_pu'] = pd.to_numeric(df_gen_pu[4])

# --- 3. 计算“联合园区”的 总负荷 和 总发电 ---
df['Gen_PV_A'] = df['PV_A_pu'] * CAP_PV_A
df['Gen_Wind_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Gen_PV_C'] = df['PV_C_pu'] * CAP_PV_C
df['Gen_Wind_C'] = df['Wind_C_pu'] * CAP_Wind_C
df['Load_Total'] = df['Load_A'] + df['Load_B'] + df['Load_C']
df['Gen_PV_Total'] = df['Gen_PV_A'] + df['Gen_PV_C']
df['Gen_Wind_Total'] = df['Gen_Wind_B'] + df['Gen_Wind_C']
df['Gen_Total'] = df['Gen_PV_Total'] + df['Gen_Wind_Total']
zero_series = pd.Series(np.zeros(24))  # (从 Q1.3 代码中移到这里)


# --- 4. 定义核心函数 (已按 critique 修正) ---
def analyze_doc_with_storage(load_series, pv_gen_series, wind_gen_series, p_cap, e_cap):
    """
    计算给定(P, E)配置下的“日运行成本(DOC)”。
    """
    p_es = p_cap;
    e_es = e_cap;
    soc_min = e_es * 0.1;
    soc_max = e_es * 0.9;
    eta = 0.95
    gen_total_series = pv_gen_series + wind_gen_series
    net_load_series = load_series - gen_total_series

    # --- “无储能”分支 (P=0 or E=0) ---
    if e_es <= 0 or p_es <= 0:
        grid_buy = np.where(net_load_series > 0, net_load_series, 0)
        curtail = np.where(net_load_series <= 0, -net_load_series, 0)
        ren_used_series = gen_total_series - curtail
        ratio_pv = np.divide(pv_gen_series, gen_total_series, out=np.zeros_like(pv_gen_series),
                             where=gen_total_series > 0)
        ratio_wind = np.divide(wind_gen_series, gen_total_series, out=np.zeros_like(wind_gen_series),
                               where=gen_total_series > 0)
        pv_used = ren_used_series * ratio_pv
        wind_used = ren_used_series * ratio_wind
        doc = (grid_buy.sum() * PRICE_GRID) + (pv_used.sum() * PRICE_PV) + (wind_used.sum() * PRICE_Wind)
        return {"doc": doc, "grid_buy": grid_buy.sum(), "curtail": curtail.sum()}

    # --- “有储能”分支 ---
    soc = np.zeros(25);
    soc[0] = soc_min
    p_charge = np.zeros(24);
    p_discharge = np.zeros(24)
    grid_buy_new = np.zeros(24);
    curtail_new = np.zeros(24)
    pv_used_new = np.zeros(24);
    wind_used_new = np.zeros(24)

    for t in range(24):
        net_load = net_load_series[t]
        if net_load > 0:
            max_can_discharge = min(p_es, (soc[t] - soc_min) * eta)
            p_discharge[t] = min(max_can_discharge, net_load)
            grid_buy_new[t] = net_load - p_discharge[t]
        elif net_load < 0:
            surplus = -net_load
            max_can_charge = min(p_es, (soc_max - soc[t]) / eta)
            p_charge[t] = min(max_can_charge, surplus)
            curtail_new[t] = surplus - p_charge[t]
        soc[t + 1] = soc[t] - (p_discharge[t] / eta) + (p_charge[t] * eta)

        pv_gen_t = pv_gen_series[t];
        wind_gen_t = wind_gen_series[t];
        gen_total_t = pv_gen_t + wind_gen_t
        ren_used_t = gen_total_t - curtail_new[t]
        if gen_total_t > 0:
            ratio_pv_t = pv_gen_t / gen_total_t; ratio_wind_t = wind_gen_t / gen_total_t
        else:
            ratio_pv_t = 0.0; ratio_wind_t = 0.0
        pv_used_new[t] = ren_used_t * ratio_pv_t;
        wind_used_new[t] = ren_used_t * ratio_wind_t

    # --- (!! 修正 !! ) ---
    # 此版本直接在 return 中求和，不定义 total_grid_buy_new 变量
    doc = (grid_buy_new.sum() * PRICE_GRID) + (pv_used_new.sum() * PRICE_PV) + (wind_used_new.sum() * PRICE_Wind)
    return {
        "doc": doc,
        "grid_buy": grid_buy_new.sum(),  # 直接使用 .sum()
        "curtail": curtail_new.sum()  # 直接使用 .sum()
    }
    # --- (!! 修正结束 !!) ---


def calculate_dic(p_cap, e_cap):
    """计算日均投资成本(DIC)"""
    return (p_cap * COST_P_ES + e_cap * COST_E_ES) / LIFESPAN_DAYS


def find_optimal_config(load_series, pv_gen_series, wind_gen_series, p_range, e_range):
    """
    执行遍历搜索，并返回“最优解”和“包含所有数据的DataFrame”
    """
    results = []
    for p in p_range:
        for e in e_range:
            sim_result = analyze_doc_with_storage(load_series, pv_gen_series, wind_gen_series, p, e)
            doc = sim_result['doc']
            dic = calculate_dic(p, e)
            tdc = doc + dic
            results.append({
                'P_cap': p, 'E_cap': e, 'TDC': tdc, 'DOC': doc, 'DIC': dic,
                'Grid_Buy': sim_result['grid_buy'],
                'Curtail': sim_result['curtail']
            })
    df_results = pd.DataFrame(results)
    optimal_config = df_results.loc[df_results['TDC'].idxmin()]

    return optimal_config, df_results


# --- 可视化函数 ---
def plot_q2_2_results(baseline_tdc, optimal_config, df_results):
    """
    生成并保存 问题 2.2 的两张核心图表
    """
    try:
        # --- 图 1: 最终方案对比柱状图 ---
        fig1, ax1 = plt.subplots(layout='constrained')
        scenarios = ['基准 (无储能)', '最优储能方案']
        costs = [baseline_tdc, optimal_config['TDC']]

        bars = ax1.bar(scenarios, costs, color=['salmon', 'forestgreen'])
        ax1.bar_label(bars, fmt='%.0f')
        ax1.set_ylabel('总成本 (TDC) (元)')
        ax1.set_title('问题2.2: 联合园区储能配置经济性对比')

        min_val = min(costs) * 0.99
        max_val = max(costs) * 1.01
        ax1.set_ylim(bottom=min_val, top=max_val)

        plt.savefig("plot_Q2_2_bar_comparison.png")
        print("\n--- 可视化图表 1/2 已生成 ---")
        print("已保存 'plot_Q2_2_bar_comparison.png'")

        # --- 图 2: 成本热力图 (Heatmap) ---
        df_pivot = df_results.pivot(index='E_cap', columns='P_cap', values='TDC')

        fig2, ax2 = plt.subplots(figsize=(10, 7), layout='constrained')
        c = ax2.pcolormesh(df_pivot.columns, df_pivot.index, df_pivot.values,
                           shading='auto', cmap='viridis_r')
        fig2.colorbar(c, ax=ax2, label='总成本 (TDC) (元)')

        ax2.plot(optimal_config['P_cap'], optimal_config['E_cap'], 'r*',
                 markersize=15, label=f"最优解 (P={optimal_config['P_cap']}, E={optimal_config['E_cap']})")

        ax2.set_title('问题2.2: 储能配置的TDC成本热力图')
        ax2.set_xlabel('储能功率 P_cap (kW)')
        ax2.set_ylabel('储能容量 E_cap (kWh)')
        ax2.legend()

        plt.savefig("plot_Q2_2_heatmap.png")
        print("\n--- 可视化图表 2/2 已生成 ---")
        print("已保存 'plot_Q2_2_heatmap.png'")

    except Exception as e:
        print(f"\n--- 可视化图表生成失败 ---")
        print(f"错误: {e}")


# --- (新增模块结束) ---


# --- 5. 主程序 ---
if __name__ == "__main__":

    print("--- 问题2(1) 联合园区 (无储能) 经济性分析 ---")
    baseline_results = analyze_doc_with_storage(
        df['Load_Total'], df['Gen_PV_Total'], df['Gen_Wind_Total'], 0, 0
    )
    baseline_tdc = baseline_results['doc']
    print(f"联合总购电量:       {baseline_results['grid_buy']:,.2f} kWh")
    print(f"联合总弃风弃光电量: {baseline_results['curtail']:,.2f} kWh")
    print(f"联合总供电成本:     {baseline_tdc:,.2f} 元")
    print(f"联合单位平均成本:   {baseline_tdc / df['Load_Total'].sum():,.4f} 元/kWh")

    print("\n\n--- 问题2(2) 联合园区 (最优储能) 配置分析 ---")
    start_time = time.time()
    P_range = np.arange(0, 301, 20)
    E_range = np.arange(0, 601, 40)
    print(f"正在执行遍历搜索 (P: {len(P_range)} 步, E: {len(E_range)} 步)...")

    optimal_combined, df_results_combined = find_optimal_config(
        df['Load_Total'],
        df['Gen_PV_Total'],
        df['Gen_Wind_Total'],
        P_range,
        E_range
    )

    print(f"搜索完成！用时 {time.time() - start_time:.2f} 秒")

    # --- 打印最终结果 ---
    print("\n" + "=" * 30)
    print("--- 联合园区 最优配置方案 ---")
    print("=" * 30)
    print(f"[基准方案 (问题2.1 无储能)]")
    print(f"总成本 TDC: {baseline_tdc:,.2f} 元")
    print(f" (详情: 购电 {baseline_results['grid_buy']:,.2f} kWh, 弃电 {baseline_results['curtail']:,.2f} kWh)")
    print(f"\n[最优配置方案 (问题2.2 有储能)]")
    print(f"最优功率: {optimal_combined['P_cap']} kW, 最优容量: {optimal_combined['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_combined['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_combined['DOC']:,.2f} 元, DIC: {optimal_combined['DIC']:,.2f} 元)")
    print(
        f" (最优方案详情: G购电 {optimal_combined['Grid_Buy']:,.2f} kWh, 弃电 {optimal_combined['Curtail']:,.2f} kWh)")
    if baseline_tdc > optimal_combined['TDC']:
        print(f"\n[经济性分析]")
        print(f"结论: 配置储能经济性改善，总成本低于无储能方案。")
        print(f"每日可节省: {baseline_tdc - optimal_combined['TDC']:,.2f} 元")
    else:
        print(f"\n[经济性分析]")
        print(f"结论: 配置储能不划算，最优方案为 P=0, E=0 (不安装储能)。")

    # --- 调用绘图函数 ---
    plot_q2_2_results(baseline_tdc, optimal_combined, df_results_combined)