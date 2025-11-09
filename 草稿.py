import pandas as pd
import numpy as np
import time
from scipy.optimize import differential_evolution

# --- 1. 定义常量 (严格忠实于 A 题 PDF) ---
# [cite: 29, 37, 62, 102]
PRICE_GRID = 1.0  # Q1/Q2 定义的网购电价 (用于峰时)
PRICE_PV_OP = 0.4  # Q1/Q2 定义的光伏运行成本 (元/kWh)
PRICE_WIND_OP = 0.5  # Q1/Q2 定义的风电运行成本 (元/kWh)

COST_PV_INV = 2500.0  # Q3 定义的光伏投资成本 (元/kW) [cite: 62]
COST_WIND_INV = 3000.0  # Q3 定义的风电投资成本 (元/kW) [cite: 62]
COST_P_ES = 800.0  # Page 1 定义的储能功率成本 (元/kW) [cite: 29]
COST_E_ES = 1800.0  # Page 1 定义的储能能量成本 (元/kWh) [cite: 29]

LIFESPAN_GEN = 5  # Q3 定义的风光回报期 (年) [cite: 62]
LIFESPAN_ES = 10  # Page 1 定义的储能寿命 (年) [cite: 30]

# (模型 [91b...] 提及的 r，我们必须假设一个)
INTEREST_RATE = 0.08  # 假设一个标准的折现率 r

# --- 问题 3(2) 的新常量 [cite: 64, 66, 67] ---
FILE_LOAD = "附件1：各园区典型日负荷数据.xlsx"
FILE_GEN_MONTHLY = "附件3：12个月各园区典型日风光发电数据.xlsx"

# 分时电价 (TOU) 数组 (0点到23点) [cite: 67]
TOU_PRICE_ARRAY = np.full(24, 0.4)  # 谷时 (其余时段)
TOU_PRICE_ARRAY[7:23] = 1.0  # 峰时 (7:00-22:00)

# 每月天数
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# --- 2. 加载和准备数据 ---
try:
    df_load_raw = pd.read_excel(FILE_LOAD)
    df_load = pd.DataFrame()
    # 负荷增长 50% [cite: 62]
    df_load['A'] = df_load_raw['园区A负荷(kW)'] * 1.5
    df_load['B'] = df_load_raw['园区B负荷(kW)'] * 1.5
    df_load['C'] = df_load_raw['园区C负荷(kW)'] * 1.5

    # 附件3 [cite: 70]
    df_gen_monthly_raw = pd.read_excel(FILE_GEN_MONTHLY, header=None, skiprows=4)

except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

zero_series = pd.Series(np.zeros(24))


# --- 3. 核心函数 ---

def capital_recovery_factor(r, n):
    """ 计算年金现值 (资本回收系数) (对应模型 [91b...]) """
    if r == 0: return 1 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


# --- *** 关键修正： analyze_doc_with_tou (忠实于模型 [d61...]) *** ---
def analyze_doc_with_tou(load_series, pv_gen_series, wind_gen_series, p_cap, e_cap):
    """
    (模型五的核心)
    在“分时电价”下计算日运行成本(DOC) - 采用修正后的正确套利逻辑
    """
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
                grid_buy_cost += net_load * TOU_PRICE_ARRAY[t]  # [cite: 67]
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

        # 严格按照题目，弃电成本为0 (没有惩罚)
        doc = grid_buy_cost + (pv_used.sum() * PRICE_PV_OP) + (wind_used.sum() * PRICE_WIND_OP)
        return doc

    # --- “有储能”分支 (套利逻辑) ---
    soc = np.zeros(25);
    soc[0] = soc_min
    grid_buy_cost = 0.0  # 从电网购买的总 *成本*
    pv_used_cost = 0.0  # 使用光伏的总 *成本*
    wind_used_cost = 0.0  # 使用风电的总 *成本*
    curtail_new = np.zeros(24)

    for t in range(24):
        # 1. 初始化本小时数据
        p_load = load_series[t]
        p_pv = pv_gen_series[t]
        p_wind = wind_gen_series[t]
        p_ren_total = p_pv + p_wind
        price = TOU_PRICE_ARRAY[t]

        p_charge = 0.0
        p_discharge = 0.0

        # 2. 优先：可再生能源满足负荷
        ren_to_load = min(p_load, p_ren_total)
        p_load_remaining = p_load - ren_to_load  # 负荷剩余缺口
        p_ren_surplus = p_ren_total - ren_to_load  # 可再生能源剩余

        if p_ren_total > 0:
            pv_used_cost += (ren_to_load * (p_pv / p_ren_total)) * PRICE_PV_OP
            wind_used_cost += (ren_to_load * (p_wind / p_ren_total)) * PRICE_WIND_OP

        # 3. 储能决策 (套利)

        # 3.1 峰时 (Price = 1.0): 优先放电
        if price == 1.0:
            if p_load_remaining > 0:  # 负荷仍有缺口
                max_can_discharge = min(p_es, (soc[t] - soc_min) * eta)
                p_discharge = min(max_can_discharge, p_load_remaining)
                p_load_remaining -= p_discharge  # 储能满足部分负荷

        # 3.2 谷时 (Price = 0.4): 优先充电
        elif price == 0.4:
            # 3.2.1 谷时 + 弃电: 优先用免费的弃电充
            if p_ren_surplus > 0:
                max_can_charge_power = min(p_es, (soc_max - soc[t]) / eta)
                p_charge_from_surplus = min(max_can_charge_power, p_ren_surplus)
                p_charge += p_charge_from_surplus
                p_ren_surplus -= p_charge_from_surplus

            # 3.2.2 谷时 + 谷电套利: 再用便宜的谷电充
            remaining_charge_power = p_es - p_charge
            remaining_charge_capacity = (soc_max - (soc[t] + p_charge * eta)) / eta

            if remaining_charge_power > 0 and remaining_charge_capacity > 0:
                p_charge_from_grid = min(remaining_charge_power, remaining_charge_capacity)
                p_charge += p_charge_from_grid
                # 充电会增加负荷
                p_load_remaining += p_charge_from_grid

                # 3.3 平时 (Price = 1.0, 但无负荷): 也要充免费的弃电
        if p_ren_surplus > 0 and p_charge < p_es:
            remaining_charge_power = p_es - p_charge
            remaining_charge_capacity = (soc_max - (soc[t] + p_charge * eta)) / eta
            p_charge_from_surplus = min(remaining_charge_power, remaining_charge_capacity, p_ren_surplus)
            p_charge += p_charge_from_surplus
            p_ren_surplus -= p_charge_from_surplus

        # 4. 结算
        curtail_new[t] = p_ren_surplus
        if p_load_remaining > 0:
            grid_buy_cost += p_load_remaining * price
        soc[t + 1] = soc[t] - (p_discharge / eta) + (p_charge * eta)

    # --- 循环结束 ---
    doc = grid_buy_cost + pv_used_cost + wind_used_cost
    return doc


# --- *** 修正结束 *** ---


def calculate_annual_cost(pv_cap, w_cap, p_es, e_es, doc_daily):
    """ (对应模型 [469...], [91b...]) """
    # 1. 年运行成本 (Cop)
    cost_op_annual = doc_daily * 365

    # 2. 年化投资成本 (Cinv)
    crf_gen = capital_recovery_factor(INTEREST_RATE, LIFESPAN_GEN)  # R1 (n=5)
    crf_es = capital_recovery_factor(INTEREST_RATE, LIFESPAN_ES)  # R2 (n=10)

    cost_inv_gen = (pv_cap * COST_PV_INV + w_cap * COST_WIND_INV)
    cost_inv_es = (p_es * COST_P_ES + e_es * COST_E_ES)

    cost_inv_annual = (cost_inv_gen * crf_gen) + (cost_inv_es * crf_es)

    # 3. 年总成本 (F)
    total_annual_cost = cost_op_annual + cost_inv_annual

    return total_annual_cost, cost_op_annual, cost_inv_annual


# --- 4. 核心：DE 算法的目标函数 (对应模型 [6ed...]) ---
def objective_function_monthly(x, park_code, load_data_monthly, gen_data_monthly_raw):
    """
    x 是决策变量 [P_pv, P_w, P_bat, E_bat]
    """
    pv_cap, w_cap, p_es, e_es = x

    if p_es < 1 or e_es < 1:
        p_es, e_es = 0, 0

    total_annual_op_cost = 0

    # --- 循环 12 个月 (对应模型 [af4...]) ---
    for m_idx, month in enumerate(range(1, 13)):
        days = DAYS_IN_MONTH[m_idx]

        # 1. 提取当月负荷
        current_load = load_data_monthly[park_code]

        # 2. 提取当月风光PU数据 (根据附件3  的列顺序)
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

        # 3. 计算当月实际发电量
        current_pv_gen = pv_pu * pv_cap
        current_wind_gen = wind_pu * w_cap

        # 4. 运行“套利”模拟，得到当月典型日 DOC
        doc_daily_m = analyze_doc_with_tou(
            current_load, current_pv_gen, current_wind_gen, p_es, e_es
        )

        # 5. 累加年运行成本
        total_annual_op_cost += doc_daily_m * days
    # --- 12个月循环结束 ---

    # 6. 计算年化投资成本 (C_inv)
    crf_gen = capital_recovery_factor(INTEREST_RATE, LIFESPAN_GEN)
    crf_es = capital_recovery_factor(INTEREST_RATE, LIFESPAN_ES)
    cost_inv_gen = (pv_cap * COST_PV_INV + w_cap * COST_WIND_INV)
    cost_inv_es = (p_es * COST_P_ES + e_es * COST_E_ES)
    cost_inv_annual = (cost_inv_gen * crf_gen) + (cost_inv_es * crf_es)

    # 7. 返回总目标：F = Cop + Cinv
    F_total = total_annual_op_cost + cost_inv_annual

    return F_total


# --- 5. 主程序 ---
if __name__ == "__main__":
    print("--- 问题3(2) 考虑12个月和分时电价的协调配置 (忠实于A题参数) ---")
    total_start_time = time.time()

    # --- 搜索边界 (Bounds) ---
    # [P_pv, P_w, P_bat, E_bat]
    bounds_A = [(0, 1500), (0, 0), (0, 300), (0, 600)]  # A园区: 无风电
    bounds_B = [(0, 0), (0, 2000), (0, 300), (0, 600)]  # B园区: 无光伏
    bounds_C = [(0, 1500), (0, 1500), (0, 300), (0, 600)]  # C园区: 都有

    # --- DE 算法的超参数 ---
    de_params = {
        'popsize': 15,
        'maxiter': 100,  # (为节省时间，设为100)
        'workers': -1,
        'updating': 'deferred',
        'disp': True,
        'tol': 0.01
    }

    # --------------------------------
    # (1) 独立运营 (逐个优化)
    # --------------------------------
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

    # --------------------------------
    # (2) 打印结果
    # --------------------------------
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