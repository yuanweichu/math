import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt  # 1. 导入图表库

# --- (新增) 设置 Matplotlib 中文字体 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
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

# --- 3. 计算所有基础数据 ---
df['Gen_A'] = df['PV_A_pu'] * CAP_PV_A
df['Net_Load_A'] = df['Load_A'] - df['Gen_A']
df['Gen_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Net_Load_B'] = df['Load_B'] - df['Gen_B']
df['Gen_PV_C'] = df['PV_C_pu'] * CAP_PV_C
df['Gen_Wind_C'] = df['Wind_C_pu'] * CAP_Wind_C
df['Gen_C_Total'] = df['Gen_PV_C'] + df['Gen_Wind_C']
df['Net_Load_C'] = df['Load_C'] - df['Gen_C_Total']

zero_series = pd.Series(np.zeros(24))

# 联合园区数据
df['Load_Total'] = df['Load_A'] + df['Load_B'] + df['Load_C']
df['Gen_PV_Total'] = df['Gen_A'] + df['Gen_PV_C']
df['Gen_Wind_Total'] = df['Gen_B'] + df['Gen_Wind_C']
df['Gen_Total'] = df['Gen_PV_Total'] + df['Gen_Wind_Total']


# --- 4. 核心函数 (!! 已替换为最终修正版 !!) ---
def analyze_doc_with_storage(load_series, pv_gen_series, wind_gen_series, p_cap, e_cap):
    """
    计算给定(P, E)配置下的“日运行成本(DOC)”。
    此版本使用“逐时成本核算”，忠实于模型。
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
        grid_buy = np.zeros(24)
        curtail = np.zeros(24);
        pv_used = np.zeros(24);
        wind_used = np.zeros(24)
        for t in range(24):
            net_load = net_load_series[t]
            if net_load > 0:
                grid_buy[t] = net_load
            else:
                curtail[t] = -net_load
            ren_used_t = gen_total_series[t] - curtail[t]
            pv_gen_t = pv_gen_series[t];
            wind_gen_t = wind_gen_series[t];
            gen_total_t = pv_gen_t + wind_gen_t
            if gen_total_t > 0:
                ratio_pv_t = pv_gen_t / gen_total_t; ratio_wind_t = wind_gen_t / gen_total_t
            else:
                ratio_pv_t = 0.0; ratio_wind_t = 0.0
            pv_used[t] = ren_used_t * ratio_pv_t;
            wind_used[t] = ren_used_t * ratio_wind_t
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
    doc = (grid_buy_new.sum() * PRICE_GRID) + (pv_used_new.sum() * PRICE_PV) + (wind_used_new.sum() * PRICE_Wind)
    return {
        "doc": doc,
        "grid_buy": grid_buy_new.sum(),
        "curtail": curtail_new.sum()
    }


# --- (!! 修正结束 !!) ---

def calculate_dic(p_cap, e_cap):
    """计算日均投资成本(DIC)"""
    return (p_cap * COST_P_ES + e_cap * COST_E_ES) / LIFESPAN_DAYS


def find_optimal_config(load_series, pv_gen_series, wind_gen_series, p_range, e_range):
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
    return optimal_config


# --- (!! 新增模块 !!) 可视化函数 ---
def plot_q2_3_comparison(ind_A, ind_B, ind_C, joint):
    """
    生成并保存 问题 2.3 的对比图表
    """
    try:
        # 1. 汇总数据
        ind_total_tdc = ind_A['TDC'] + ind_B['TDC'] + ind_C['TDC']
        joint_total_tdc = joint['TDC']

        ind_total_E = ind_A['E_cap'] + ind_B['E_cap'] + ind_C['E_cap']
        joint_total_E = joint['E_cap']

        ind_total_buy = ind_A['Grid_Buy'] + ind_B['Grid_Buy'] + ind_C['Grid_Buy']
        joint_total_buy = joint['Grid_Buy']

        ind_total_curtail = ind_A['Curtail'] + ind_B['Curtail'] + ind_C['Curtail']
        joint_total_curtail = joint['Curtail']

        # --- 图 1: 总成本对比 (回答“有何经济收益”) ---
        fig1, ax1 = plt.subplots(layout='constrained')
        scenarios = ['独立运营 (总计)', '联合运营']
        costs = [ind_total_tdc, joint_total_tdc]

        bars = ax1.bar(scenarios, costs, color=['coral', 'mediumseagreen'])
        ax1.bar_label(bars, fmt='%.0f')
        ax1.set_ylabel('最优总成本 (TDC) (元)')
        ax1.set_title('问题2.3: 独立运营 vs 联合运营 总成本对比')

        min_val = min(costs) * 0.98  # 放大
        max_val = max(costs) * 1.02
        ax1.set_ylim(bottom=min_val, top=max_val)

        plt.savefig("plot_Q2_3_cost_comparison.png")
        print("\n--- 可视化图表 1/2 已生成 ---")
        print("已保存 'plot_Q2_3_cost_comparison.png'")

        # --- 图 2: 因素分析 (回答“为何改变”) ---
        fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
        fig2.suptitle('问题2.3: 导致经济收益改变的因素分析', fontsize=16)

        # 子图 2a: 规模经济性 (储能容量)
        data_E = [ind_total_E, joint_total_E]
        ax2.bar(scenarios, data_E, color=['coral', 'mediumseagreen'])
        ax2.set_title('规模经济性: 储能容量 (kWh)')
        ax2.bar_label(ax2.containers[0], fmt='%.0f')

        # 子图 2b: 时空互补性 (购电量)
        data_buy = [ind_total_buy, joint_total_buy]
        ax3.bar(scenarios, data_buy, color=['coral', 'mediumseagreen'])
        ax3.set_title('时空互补性: 总购电量 (kWh)')
        ax3.bar_label(ax3.containers[0], fmt='%.0f')

        # 子图 2c: 时空互补性 (弃电量)
        data_curtail = [ind_total_curtail, joint_total_curtail]
        ax4.bar(scenarios, data_curtail, color=['coral', 'mediumseagreen'])
        ax4.set_title('时空互补性: 总弃电量 (kWh)')
        ax4.bar_label(ax4.containers[0], fmt='%.0f')

        plt.savefig("plot_Q2_3_factors_comparison.png")
        print("\n--- 可视化图表 2/2 已生成 ---")
        print("已保存 'plot_Q2_3_factors_comparison.png'")

    except Exception as e:
        print(f"\n--- 可视化图表生成失败 ---")
        print(f"错误: {e}")


# --- (新增模块结束) ---


# --- 5. 主程序 ---
if __name__ == "__main__":

    print("--- 正在执行 问题1(3) [独立运营最优解] ... ---")
    P_range_ind = np.arange(0, 151, 10)
    E_range_ind = np.arange(0, 301, 20)

    start_A = time.time()
    optimal_A = find_optimal_config(df['Load_A'], df['Gen_A'], zero_series, P_range_ind, E_range_ind)
    print(f"园区A 优化完成 (用时 {time.time() - start_A:.2f} s)")

    start_B = time.time()
    optimal_B = find_optimal_config(df['Load_B'], zero_series, df['Gen_B'], P_range_ind, E_range_ind)
    print(f"园区B 优化完成 (用时 {time.time() - start_B:.2f} s)")

    start_C = time.time()
    optimal_C = find_optimal_config(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], P_range_ind, E_range_ind)
    print(f"园区C 优化完成 (用时 {time.time() - start_C:.2f} s)")

    tdc_independent_total = optimal_A['TDC'] + optimal_B['TDC'] + optimal_C['TDC']

    print("\n--- 正在执行 问题2(2) [联合运营最优解] ... ---")
    P_range_joint = np.arange(0, 301, 20)
    E_range_joint = np.arange(0, 601, 40)

    start_J = time.time()
    optimal_joint = find_optimal_config(
        df['Load_Total'],
        df['Gen_PV_Total'],
        df['Gen_Wind_Total'],
        P_range_joint,
        E_range_joint
    )
    print(f"联合园区 优化完成 (用时 {time.time() - start_J:.2f} s)")
    tdc_joint_total = optimal_joint['TDC']

    # ----------------------------------------------------
    # (3) 执行 问题2(3) [对比模型]
    # ----------------------------------------------------
    print("\n\n" + "=" * 40)
    print("--- 问题2(3) 经济收益对比分析 ---")
    print("=" * 40)
    print("\n[独立运营 总成本 (问题1.3 汇总)]")
    print(f" 园区A最优TDC: {optimal_A['TDC']:,.2f} 元 (P={optimal_A['P_cap']} kW, E={optimal_A['E_cap']} kWh)")
    print(f" 园区B最优TDC: {optimal_B['TDC']:,.2f} 元 (P={optimal_B['P_cap']} kW, E={optimal_B['E_cap']} kWh)")
    print(f" 园区C最优TDC: {optimal_C['TDC']:,.2f} 元 (P={optimal_C['P_cap']} kW, E={optimal_C['E_cap']} kWh)")
    print(f"----------------------------------------")
    print(f" **独立运营总成本合计: {tdc_independent_total:,.2f} 元**")
    print("\n[联合运营 总成本 (问题2.2)]")
    print(
        f" 园区联合最优TDC: {optimal_joint['TDC']:,.2f} 元 (P={optimal_joint['P_cap']} kW, E={optimal_joint['E_cap']} kWh)")
    print(f"----------------------------------------")
    print(f" **联合运营总成本合计: {tdc_joint_total:,.2f} 元**")
    print("\n[经济收益分析 (模型三)]")
    economic_benefit = tdc_independent_total - tdc_joint_total
    if economic_benefit > 0:
        print(f"**总经济收益 (ΔC): {economic_benefit:,.2f} 元**")
        print("结论：联合运营比独立运营更经济，每日可节省成本。")
    else:
        print(f"**总经济收益 (ΔC): {economic_benefit:,.2f} 元**")
        print("结论：联合运营不经济。")
    print("\n[导致经济收益改变的主要因素分析]...")

    # --- (新增) 调用绘图函数 ---
    plot_q2_3_comparison(optimal_A, optimal_B, optimal_C, optimal_joint)