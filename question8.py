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


# --- 1. 定义常量 (严格忠实于 A 题 PDF) ---
PRICE_GRID = 1.0
PRICE_PV_OP = 0.4
PRICE_WIND_OP = 0.5
COST_PV_INV = 2500.0
COST_WIND_INV = 3000.0
COST_P_ES = 800.0
COST_E_ES = 1800.0
LIFESPAN_GEN = 5
LIFESPAN_ES = 10
INTEREST_RATE = 0.08
FILE_LOAD = "附件1：各园区典型日负荷数据.xlsx"
FILE_GEN_MONTHLY = "附件3：12个月各园区典型日风光发电数据.xlsx"
TOU_PRICE_ARRAY = np.full(24, 0.4)
TOU_PRICE_ARRAY[7:23] = 1.0
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# --- 2. 加载和准备数据 ---
try:
    df_load_raw = pd.read_excel(FILE_LOAD)
    df_load = pd.DataFrame()
    df_load['A'] = df_load_raw['园区A负荷(kW)'] * 1.5
    df_load['B'] = df_load_raw['园区B负荷(kW)'] * 1.5
    df_load['C'] = df_load_raw['园区C负荷(kW)'] * 1.5
    df_gen_monthly_raw = pd.read_excel(FILE_GEN_MONTHLY, header=None, skiprows=4)
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()
zero_series = pd.Series(np.zeros(24))


# --- 3. 核心函数 ---

def capital_recovery_factor(r, n):
    """ 计算年金现值 (资本回收系数) """
    if r == 0: return 1 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def analyze_doc_with_tou(load_series, pv_gen_series, wind_gen_series, p_cap, e_cap):
    """ (已修正) 在“分时电价”下计算日运行成本(DOC) """
    p_es = p_cap;
    e_es = e_cap;
    soc_min = e_es * 0.1;
    soc_max = e_es * 0.9;
    eta = 0.95
    gen_total_series = pv_gen_series + wind_gen_series

    # --- “无储能”分支 (P=0 or E=0) ---
    if e_es <= 0 or p_es <= 0:
        net_load_series = load_series - gen_total_series
        grid_buy_cost = 0.0
        curtail = np.zeros(24);
        pv_used = np.zeros(24);
        wind_used = np.zeros(24)
        for t in range(24):
            net_load = net_load_series[t]
            if net_load > 0:
                grid_buy_cost += net_load * TOU_PRICE_ARRAY[t]
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
        doc = grid_buy_cost + (pv_used.sum() * PRICE_PV_OP) + (wind_used.sum() * PRICE_WIND_OP)
        return doc

    # --- “有储能”分支 (套利逻辑) ---
    soc = np.zeros(25);
    soc[0] = soc_min
    grid_buy_cost = 0.0;
    pv_used_cost = 0.0;
    wind_used_cost = 0.0
    curtail_new = np.zeros(24)

    for t in range(24):
        p_load = load_series[t];
        p_pv = pv_gen_series[t];
        p_wind = wind_gen_series[t]
        p_ren_total = p_pv + p_wind;
        price = TOU_PRICE_ARRAY[t]
        p_charge = 0.0;
        p_discharge = 0.0

        ren_to_load = min(p_load, p_ren_total)
        p_load_remaining = p_load - ren_to_load
        p_ren_surplus = p_ren_total - ren_to_load
        if p_ren_total > 0:
            pv_used_cost += (ren_to_load * (p_pv / p_ren_total)) * PRICE_PV_OP
            wind_used_cost += (ren_to_load * (p_wind / p_ren_total)) * PRICE_WIND_OP

        if price == 1.0:
            if p_load_remaining > 0:
                max_can_discharge = min(p_es, (soc[t] - soc_min) * eta)
                p_discharge = min(max_can_discharge, p_load_remaining)
                p_load_remaining -= p_discharge
        elif price == 0.4:
            if p_ren_surplus > 0:
                max_can_charge_power = min(p_es, (soc_max - soc[t]) / eta)
                p_charge_from_surplus = min(max_can_charge_power, p_ren_surplus)
                p_charge += p_charge_from_surplus
                p_ren_surplus -= p_charge_from_surplus
            remaining_charge_power = p_es - p_charge
            remaining_charge_capacity = (soc_max - (soc[t] + p_charge * eta)) / eta
            if remaining_charge_power > 0 and remaining_charge_capacity > 0:
                p_charge_from_grid = min(remaining_charge_power, remaining_charge_capacity)
                p_charge += p_charge_from_grid
                p_load_remaining += p_charge_from_grid
        if p_ren_surplus > 0 and p_charge < p_es:
            remaining_charge_power = p_es - p_charge
            remaining_charge_capacity = (soc_max - (soc[t] + p_charge * eta)) / eta
            p_charge_from_surplus = min(remaining_charge_power, remaining_charge_capacity, p_ren_surplus)
            p_charge += p_charge_from_surplus
            p_ren_surplus -= p_charge_from_surplus

        curtail_new[t] = p_ren_surplus
        if p_load_remaining > 0:
            grid_buy_cost += p_load_remaining * price
        soc[t + 1] = soc[t] - (p_discharge / eta) + (p_charge * eta)

    doc = grid_buy_cost + pv_used_cost + wind_used_cost
    return doc


# --- (函数定义结束) ---


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
def objective_function_monthly(x, park_code, load_data_monthly, gen_data_monthly_raw):
    """ x 是决策变量 [P_pv, P_w, P_bat, E_bat] """
    pv_cap, w_cap, p_es, e_es = x

    if p_es < 1 or e_es < 1:
        p_es, e_es = 0, 0

    total_annual_op_cost = 0

    # --- 循环 12 个月 ---
    for m_idx, month in enumerate(range(1, 13)):
        days = DAYS_IN_MONTH[m_idx]
        current_load = load_data_monthly[park_code]

        col_idx_base = (m_idx * 4) + 1
        col_A_pv = col_idx_base
        col_B_w = col_idx_base + 1
        col_C_w = col_idx_base + 2
        col_C_pv = col_idx_base + 3

        if park_code == 'A':
            pv_pu = pd.to_numeric(gen_data_monthly_raw[col_A_pv])
            wind_pu = zero_series
        elif park_code == 'B':
            pv_pu = zero_series
            wind_pu = pd.to_numeric(gen_data_monthly_raw[col_B_w])
        elif park_code == 'C':
            pv_pu = pd.to_numeric(gen_data_monthly_raw[col_C_pv])
            wind_pu = pd.to_numeric(gen_data_monthly_raw[col_C_w])

        current_pv_gen = pv_pu * pv_cap
        current_wind_gen = wind_pu * w_cap

        doc_daily_m = analyze_doc_with_tou(
            current_load, current_pv_gen, current_wind_gen, p_es, e_es
        )
        total_annual_op_cost += doc_daily_m * days
    # --- 12个月循环结束 ---

    # (计算年化投资成本)
    crf_gen = capital_recovery_factor(INTEREST_RATE, LIFESPAN_GEN)
    crf_es = capital_recovery_factor(INTEREST_RATE, LIFESPAN_ES)
    cost_inv_gen = (pv_cap * COST_PV_INV + w_cap * COST_WIND_INV)
    cost_inv_es = (p_es * COST_P_ES + e_es * COST_E_ES)
    cost_inv_annual = (cost_inv_gen * crf_gen) + (cost_inv_es * crf_es)

    F_total = total_annual_op_cost + cost_inv_annual

    return F_total


# --- (!! 已修改 !!) 可视化函数 ---
def plot_q3_2_results(results):
    """
    生成并保存 问题 3(2) 的最终结果图表
    """
    try:
        parks = ['园区A', '园区B', '园区C']

        # 1. 提取数据
        costs_F = [results['A']['F'], results['B']['F'], results['C']['F']]
        configs_pv = [results['A']['x'][0], results['B']['x'][0], results['C']['x'][0]]
        configs_w = [results['A']['x'][1], results['B']['x'][1], results['C']['x'][1]]
        configs_p_bat = [results['A']['x'][2], results['B']['x'][2], results['C']['x'][2]]
        configs_e_bat = [results['A']['x'][3], results['B']['x'][3], results['C']['x'][3]]

        # 2. 计算成本构成
        costs_Op = []
        costs_Inv = []
        for park_code, result in results.items():
            (pv, w, p_bat, e_bat) = result['x']

            # (重新计算 DOC)
            total_annual_op_cost = 0
            for m_idx in range(12):
                days = DAYS_IN_MONTH[m_idx]
                current_load = df_load[park_code]
                col_idx_base = (m_idx * 4) + 1
                if park_code == 'A':
                    pv_pu = pd.to_numeric(df_gen_monthly_raw[col_idx_base]);
                    wind_pu = zero_series
                elif park_code == 'B':
                    pv_pu = zero_series;
                    wind_pu = pd.to_numeric(df_gen_monthly_raw[col_idx_base + 1])
                elif park_code == 'C':
                    pv_pu = pd.to_numeric(df_gen_monthly_raw[col_idx_base + 3]);
                    wind_pu = pd.to_numeric(df_gen_monthly_raw[col_idx_base + 2])

                current_pv_gen = pv_pu * pv
                current_wind_gen = wind_pu * w
                doc_daily_m = analyze_doc_with_tou(current_load, current_pv_gen, current_wind_gen, p_bat, e_bat)
                total_annual_op_cost += doc_daily_m * days

            # (重新计算 DIC)
            crf_gen = capital_recovery_factor(INTEREST_RATE, LIFESPAN_GEN)
            crf_es = capital_recovery_factor(INTEREST_RATE, LIFESPAN_ES)
            cost_inv_gen = (pv * COST_PV_INV + w * COST_WIND_INV)
            cost_inv_es = (p_bat * COST_P_ES + e_bat * COST_E_ES)
            cost_inv_annual = (cost_inv_gen * crf_gen) + (cost_inv_es * crf_es)

            costs_Op.append(total_annual_op_cost)
            costs_Inv.append(cost_inv_annual)

        # --- 图 1: 年总成本构成 (堆叠柱状图) ---
        fig1, ax1 = plt.subplots(figsize=(10, 7), layout='constrained')
        width = 0.5
        rects1 = ax1.bar(parks, costs_Inv, width, label='年化投资成本 (C_inv)', color='royalblue')
        rects2 = ax1.bar(parks, costs_Op, width, bottom=costs_Inv,
                         label='年运行成本 (C_op)', color='sandybrown')
        ax1.set_ylabel('年总成本 (F) (元)')
        ax1.set_title('问题3(2): 各园区最优配置的年总成本构成')
        ax1.legend()
        for i, rect in enumerate(rects2):
            total_height = rect.get_height() + costs_Inv[i]
            ax1.text(rect.get_x() + rect.get_width() / 2, total_height,
                     f'总计: {total_height:,.0f}', ha='center', va='bottom', fontsize=9, weight='bold')

        plt.savefig("plot_Q3_2_Cost_Composition.png")
        print("\n--- 可视化图表 1/3 已生成 ---")
        print("已保存 'plot_Q3_2_Cost_Composition.png'")

        # --- (!! 修正后的图 2 !!)：风光配置对比 ---
        fig2, ax2 = plt.subplots(figsize=(10, 7), layout='constrained')
        labels_gen = ['光伏 P_pv (kW)', '风电 P_w (kW)']
        x_gen = np.arange(len(labels_gen))
        width_gen = 0.25
        multiplier_gen = -1

        data_gen = {
            '园区A': [configs_pv[0], configs_w[0]],
            '园区B': [configs_pv[1], configs_w[1]],
            '园区C': [configs_pv[2], configs_w[2]],
        }

        for park, config in data_gen.items():
            offset = width_gen * multiplier_gen
            rects = ax2.bar(x_gen + offset, config, width_gen, label=park)
            ax2.bar_label(rects, padding=3, fmt='%.0f', rotation=90, fontsize=8)
            multiplier_gen += 1

        ax2.set_ylabel('最优配置容量 (kW)')
        ax2.set_title('问题3(2): 各园区最优【风光】配置方案对比')
        ax2.set_xticks(x_gen, labels_gen)
        ax2.legend(loc='upper left', ncols=3)
        ax2.set_ylim(top=ax2.get_ylim()[1] * 1.15)

        plt.savefig("plot_Q3_2_Generation_Config.png")
        print("\n--- 可视化图表 2/3 已生成 ---")
        print("已保存 'plot_Q3_2_Generation_Config.png'")

        # --- (!! 修正后的图 3 !!)：储能配置对比 ---
        fig3, ax3 = plt.subplots(figsize=(10, 7), layout='constrained')
        labels_es = ['储能 P_bat (kW)', '储能 E_bat (kWh)']
        x_es = np.arange(len(labels_es))
        width_es = 0.25
        multiplier_es = -1

        data_es = {
            '园区A': [configs_p_bat[0], configs_e_bat[0]],
            '园区B': [configs_p_bat[1], configs_e_bat[1]],
            '园区C': [configs_p_bat[2], configs_e_bat[2]],
        }

        for park, config in data_es.items():
            offset = width_es * multiplier_es
            rects = ax3.bar(x_es + offset, config, width_es, label=park)
            ax3.bar_label(rects, padding=3, fmt='%.1f', rotation=90, fontsize=8)  # (使用 .1f 显示小数)
            multiplier_es += 1

        ax3.set_ylabel('最优配置容量 (kW / kWh)')
        ax3.set_title('问题3(2): 各园区最优【储能】配置方案对比')
        ax3.set_xticks(x_es, labels_es)
        ax3.legend(loc='upper left', ncols=3)
        ax3.set_ylim(top=ax3.get_ylim()[1] * 1.15)  # 增加顶部空间

        plt.savefig("plot_Q3_2_Storage_Config.png")
        print("\n--- 可视化图表 3/3 已生成 ---")
        print("已保存 'plot_Q3_2_Storage_Config.png'")

    except Exception as e:
        print(f"\n--- 可视化图表生成失败 ---")
        print(f"错误: {e}")


# --- (新增模块结束) ---


# --- 5. 主程序 ---
if __name__ == "__main__":
    print("--- 问题3(2) 考虑12个月和分时电价的协调配置 (忠实于A题参数) ---")
    total_start_time = time.time()

    bounds_A = [(0, 1500), (0, 0), (0, 300), (0, 600)]
    bounds_B = [(0, 0), (0, 2000), (0, 300), (0, 600)]
    bounds_C = [(0, 1500), (0, 1500), (0, 300), (0, 600)]

    de_params = {'popsize': 15, 'maxiter': 100, 'workers': -1,
                 'updating': 'deferred', 'disp': True, 'tol': 0.01}

    # (1) 独立运营 (逐个优化)
    print("\n--- 正在优化 [独立运营] 方案 ---")
    print("\n[园区A] (DE 运行中...)")
    result_A = differential_evolution(objective_function_monthly, bounds_A,
                                      args=('A', df_load, df_gen_monthly_raw),
                                      **de_params)
    print("\n[园区B] (DE 运行中...)")
    result_B = differential_evolution(objective_function_monthly, bounds_B,
                                      args=('B', df_load, df_gen_monthly_raw),
                                      **de_params)
    print("\n[园区C] (DE 运行中...)")
    result_C = differential_evolution(objective_function_monthly, bounds_C,
                                      args=('C', df_load, df_gen_monthly_raw),
                                      **de_params)

    # (2) 打印结果
    print("\n\n" + "=" * 30)
    print("--- 问题3(2) 最终配置结果 ---")
    print("=" * 30)

    (pv_A, w_A, p_bat_A, e_bat_A) = result_A.x
    F_A = result_A.fun
    (pv_B, w_B, p_bat_B, e_bat_B) = result_B.x
    F_B = result_B.fun
    (pv_C, w_C, p_bat_C, e_bat_C) = result_C.x
    F_C = result_C.fun

    print(f"\n[独立运营 最优配置] (总用时: {time.time() - total_start_time:.2f} s)")
    print(f" 园区A: P_pv={pv_A:.2f} kW, P_w={w_A:.2f} kW, P_bat={p_bat_A:.2f} kW, E_bat={e_bat_A:.2f} kWh")
    print(f"      年总成本 (F_A): {F_A:,.2f} 元")
    print(f" 园区B: P_pv={pv_B:.2f} kW, P_w={w_B:.2f} kW, P_bat={p_bat_B:.2f} kW, E_bat={e_bat_B:.2f} kWh")
    print(f"      年总成本 (F_B): {F_B:,.2f} 元")
    print(f" 园区C: P_pv={pv_C:.2f} kW, P_w={w_C:.2f} kW, P_bat={p_bat_C:.2f} kW, E_bat={e_bat_C:.2f} kWh")
    print(f"      年总成本 (F_C): {F_C:,.2f} 元")

    # --- (新增) 调用绘图函数 ---
    all_results = {
        'A': {'x': result_A.x, 'F': result_A.fun},
        'B': {'x': result_B.x, 'F': result_B.fun},
        'C': {'x': result_C.x, 'F': result_C.fun}
    }
    plot_q3_2_results(all_results)