import pandas as pd
import numpy as np
import time
from scipy.optimize import differential_evolution  # 导入智能算法
import matplotlib.pyplot as plt  # 1. 导入图表库

# --- (新增) 设置 Matplotlib 中文字体 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置失败 (可能未安装 'SimHei')，图表标签可能显示异常: {e}")
# --- 结束新增 ---


# --- 1. 定义常量 ---
# (注意：价格均来自 A 题 PDF)
PRICE_GRID = 1.0
PRICE_PV_OP = 0.4  # 光伏 *运行* 成本
PRICE_WIND_OP = 0.5  # 风电 *运行* 成本

COST_PV_INV = 2500.0  # 光伏 *投资* 成本 (元/kW)
COST_WIND_INV = 3000.0  # 风电 *投资* 成本 (元/kW)
COST_P_ES = 800.0  # 储能功率 *投资* 成本 (元/kW)
COST_E_ES = 1800.0  # 储能能量 *投资* 成本 (元/kWh)

LIFESPAN_GEN = 5  # 风光回报期 (年)
LIFESPAN_ES = 10  # 储能寿命 (年)
INTEREST_RATE = 0.08  # 假设一个标准的折现率 r

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
# 负荷增长 50%
df['Load_A'] = df_load['园区A负荷(kW)'] * 1.5
df['Load_B'] = df_load['园区B负荷(kW)'] * 1.5
df['Load_C'] = df_load['园区C负荷(kW)'] * 1.5

df['PV_A_pu'] = pd.to_numeric(df_gen_pu[1])
df['Wind_B_pu'] = pd.to_numeric(df_gen_pu[2])
df['PV_C_pu'] = pd.to_numeric(df_gen_pu[3])
df['Wind_C_pu'] = pd.to_numeric(df_gen_pu[4])

# 准备联合园区数据
df['Load_Joint'] = df['Load_A'] + df['Load_B'] + df['Load_C']
zero_series = pd.Series(np.zeros(24))


# --- 3. 核心函数 ---
def capital_recovery_factor(r, n):
    """ 计算年金现值 (资本回收系数) """
    if r == 0: return 1 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def analyze_doc_with_storage(load_series, pv_gen_series, wind_gen_series, p_cap, e_cap):
    """ (已修正) 计算给定(P, E)配置下的“日运行成本(DOC)” """
    p_es = p_cap;
    e_es = e_cap;
    soc_min = e_es * 0.1;
    soc_max = e_es * 0.9;
    eta = 0.95
    gen_total_series = pv_gen_series + wind_gen_series
    net_load_series = load_series - gen_total_series

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
        doc = (grid_buy.sum() * PRICE_GRID) + (pv_used.sum() * PRICE_PV_OP) + (wind_used.sum() * PRICE_WIND_OP)
        return {"doc": doc, "grid_buy": grid_buy.sum(), "curtail": curtail.sum()}

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
    doc = (grid_buy_new.sum() * PRICE_GRID) + (pv_used_new.sum() * PRICE_PV_OP) + (wind_used_new.sum() * PRICE_WIND_OP)
    return {"doc": doc, "grid_buy": grid_buy_new.sum(), "curtail": curtail_new.sum()}


def calculate_annual_cost(pv_cap, w_cap, p_es, e_es, doc_daily):
    """ 计算模型的总目标函数 (年总成本 F) """
    cost_op_annual = doc_daily * 365
    crf_gen = capital_recovery_factor(INTEREST_RATE, LIFESPAN_GEN)
    crf_es = capital_recovery_factor(INTEREST_RATE, LIFESPAN_ES)
    cost_inv_gen = (pv_cap * COST_PV_INV + w_cap * COST_WIND_INV)
    cost_inv_es = (p_es * COST_P_ES + e_es * COST_E_ES)
    cost_inv_annual = (cost_inv_gen * crf_gen) + (cost_inv_es * crf_es)
    total_annual_cost = cost_op_annual + cost_inv_annual
    return total_annual_cost, cost_op_annual, cost_inv_annual


# --- 4. 核心：DE 算法的目标函数 ---
def objective_function(x, load_series, pv_pu_series, wind_pu_series):
    """ x 是决策变量 [P_pv, P_w, P_bat, E_bat] """
    pv_cap, w_cap, p_es, e_es = x
    if p_es < 1 or e_es < 1:
        p_es, e_es = 0, 0
    current_pv_gen = pv_pu_series * pv_cap
    current_wind_gen = wind_pu_series * w_cap
    sim_result = analyze_doc_with_storage(
        load_series, current_pv_gen, current_wind_gen, p_es, e_es
    )
    doc_daily = sim_result['doc']
    tdc_annual, _, _ = calculate_annual_cost(
        pv_cap, w_cap, p_es, e_es, doc_daily
    )
    return tdc_annual


# --- (!! 新增模块 !!) 可视化函数 ---
def plot_q3_1_comparison(ind_total, joint_total, ind_config, joint_config):
    """
    生成并保存 问题 3(1) 的两张对比图表
    """
    try:
        # --- 图 1: 总成本对比 (经济收益) ---
        fig1, ax1 = plt.subplots(figsize=(7, 6), layout='constrained')
        scenarios = ['独立运营 (总计)', '联合运营']
        costs = [ind_total, joint_total]

        colors = ['coral', 'mediumseagreen']
        bars = ax1.bar(scenarios, costs, color=colors, width=0.6)
        ax1.bar_label(bars, fmt='%.0f')
        ax1.set_ylabel('年总成本 (F) (元)')
        ax1.set_title('问题3(1): 独立运营 vs 联合运营 总成本对比')

        # 放大Y轴
        min_val = min(costs) * 0.98
        max_val = max(costs) * 1.02
        ax1.set_ylim(bottom=min_val, top=max_val)

        plt.savefig("plot_Q3_1_Cost_Comparison.png")
        print("\n--- 可视化图表 1/2 已生成 ---")
        print("已保存 'plot_Q3_1_Cost_Comparison.png'")

        # --- 图 2: 总配置对比 (因素分析) ---
        fig2, ax2 = plt.subplots(figsize=(10, 7), layout='constrained')

        labels = ['光伏 P_pv (kW)', '风电 P_w (kW)', '储能功率 P_bat (kW)', '储能容量 E_bat (kWh)']
        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax2.bar(x - width / 2, ind_config, width, label='独立运营 (A+B+C 总和)', color='coral')
        rects2 = ax2.bar(x + width / 2, joint_config, width, label='联合运营', color='mediumseagreen')

        ax2.set_ylabel('总配置容量')
        ax2.set_title('问题3(1): 独立 vs 联合 的最优配置对比')
        ax2.set_xticks(x, labels)
        ax2.legend()
        ax2.bar_label(rects1, padding=3, fmt='%.0f')
        ax2.bar_label(rects2, padding=3, fmt='%.0f')

        plt.savefig("plot_Q3_1_Config_Comparison.png")
        print("\n--- 可视化图表 2/2 已生成 ---")
        print("已保存 'plot_Q3_1_Config_Comparison.png'")

    except Exception as e:
        print(f"\n--- 可视化图表生成失败 ---")
        print(f"错误: {e}")


# --- (新增模块结束) ---

# --- 5. 主程序 ---
if __name__ == "__main__":

    print("--- 问题3(1) 风光储协调配置分析 (使用智能算法 DE) ---")
    total_start_time = time.time()
    bounds_A = [(0, 1500), (0, 0), (0, 300), (0, 600)]
    bounds_B = [(0, 0), (0, 2000), (0, 300), (0, 600)]
    bounds_C = [(0, 1500), (0, 1500), (0, 300), (0, 600)]
    bounds_J = [(0, 3000), (0, 3000), (0, 600), (0, 1200)]
    de_params = {'popsize': 15, 'maxiter': 100, 'workers': -1,
                 'updating': 'deferred', 'disp': True, 'tol': 0.01}

    # (1) 独立运营
    print("\n--- 正在优化 [独立运营] 方案 ---")
    print("\n[园区A] (DE 运行中...)")
    result_A = differential_evolution(objective_function, bounds_A,
                                      args=(df['Load_A'], df['PV_A_pu'], zero_series),
                                      **de_params)
    print("\n[园区B] (DE 运行中...)")
    result_B = differential_evolution(objective_function, bounds_B,
                                      args=(df['Load_B'], zero_series, df['Wind_B_pu']),
                                      **de_params)
    print("\n[园区C] (DE 运行中...)")
    result_C = differential_evolution(objective_function, bounds_C,
                                      args=(df['Load_C'], df['PV_C_pu'], df['Wind_C_pu']),
                                      **de_params)

    # (2) 联合运营
    print("\n--- 正在优化 [联合运营] 方案 ---")
    print("\n[联合园区] (DE 运行中...)")
    result_J = differential_evolution(objective_function, bounds_J,
                                      args=(df['Load_Joint'], df['PV_A_pu'], df['Wind_B_pu']),
                                      **de_params)

    # (3) 打印结果
    print("\n\n" + "=" * 30)
    print("--- 问题3(1) 协调配置结果 ---")
    print("=" * 30)
    (pv_A, w_A, p_bat_A, e_bat_A) = result_A.x
    F_A = result_A.fun
    (pv_B, w_B, p_bat_B, e_bat_B) = result_B.x
    F_B = result_B.fun
    (pv_C, w_C, p_bat_C, e_bat_C) = result_C.x
    F_C = result_C.fun
    (pv_J, w_J, p_bat_J, e_bat_J) = result_J.x
    F_Joint = result_J.fun
    F_Independent_Total = F_A + F_B + F_C

    print(f"\n[独立运营 最优配置] (总用时: {time.time() - total_start_time:.2f} s)")
    print(f" 园区A: P_pv={pv_A:.2f} kW, P_w={w_A:.2f} kW, P_bat={p_bat_A:.2f} kW, E_bat={e_bat_A:.2f} kWh")
    print(f"      年总成本 (F_A): {F_A:,.2f} 元")
    print(f" 园区B: P_pv={pv_B:.2f} kW, P_w={w_B:.2f} kW, P_bat={p_bat_B:.2f} kW, E_bat={e_bat_B:.2f} kWh")
    print(f"      年总成本 (F_B): {F_B:,.2f} 元")
    print(f" 园区C: P_pv={pv_C:.2f} kW, P_w={w_C:.2f} kW, P_bat={p_bat_C:.2f} kW, E_bat={e_bat_C:.2f} kWh")
    print(f"      年总成本 (F_C): {F_C:,.2f} 元")
    print("--------------------------------------------------")
    print(f" **独立运营总计年成本 (F_独立): {F_Independent_Total:,.2f} 元**")
    print("\n[联合运营 最优配置]")
    print(f" 联合: P_pv={pv_J:.2f} kW, P_w={w_J:.2f} kW, P_bat={p_bat_J:.2f} kW, E_bat={e_bat_J:.2f} kWh")
    print(f" **联合运营总计年成本 (F_联合): {F_Joint:,.2f} 元**")
    print("\n[对比分析]")
    if F_Independent_Total > F_Joint:
        print(f"结论：联合运营更优，每年可节省 (F_独立 - F_联合) = {F_Independent_Total - F_Joint:,.2f} 元。")
    else:
        print(f"结论：独立运营更优，联合运营成本高出 {F_Joint - F_Independent_Total:,.2f} 元。")

    # --- (新增) 调用绘图函数 ---

    # 汇总独立园区的总配置
    ind_config_sum = [
        pv_A + pv_B + pv_C,  # Total P_pv
        w_A + w_B + w_C,  # Total P_w
        p_bat_A + p_bat_B + p_bat_C,  # Total P_bat
        e_bat_A + e_bat_B + e_bat_C  # Total E_bat
    ]
    # 联合园区的配置
    joint_config = [pv_J, w_J, p_bat_J, e_bat_J]

    # 调用
    plot_q3_1_comparison(F_Independent_Total, F_Joint, ind_config_sum, joint_config)