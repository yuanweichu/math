import pandas as pd
import numpy as np
import time

# --- 1. 定义常量 ---
CAP_PV_A = 750.0
CAP_Wind_B = 1000.0
CAP_PV_C = 600.0
CAP_Wind_C = 500.0

PRICE_GRID = 1.0
PRICE_PV = 0.4
PRICE_Wind = 0.5

# 储能投资成本
COST_P_ES = 800.0  # 功率成本 (元/kW)
COST_E_ES = 1800.0  # 能量成本 (元/kWh)
LIFESPAN_DAYS = 10 * 365  # 寿命 (天)

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
df['Gen_PV_Total'] = df['Gen_A'] + df['Gen_PV_C']  # 注意 Gen_A 是 PV
df['Gen_Wind_Total'] = df['Gen_B'] + df['Gen_Wind_C']  # 注意 Gen_B 是 Wind


# --- 4. 核心函数 (复用) ---

def analyze_doc_with_storage(load_series, pv_gen_series, wind_gen_series, p_cap, e_cap):
    """
    计算给定(P, E)配置下的“日运行成本(DOC)”。
    """
    p_es = p_cap
    e_es = e_cap
    soc_min = e_es * 0.1
    soc_max = e_es * 0.9
    eta = 0.95

    net_load_series = load_series - (pv_gen_series + wind_gen_series)

    # 如果p或e为0，则执行“无储能”的DOC计算
    if e_es <= 0 or p_es <= 0:
        grid_buy = np.where(net_load_series > 0, net_load_series, 0).sum()
        curtail = np.where(net_load_series <= 0, -net_load_series, 0).sum()

        gen_total = pv_gen_series.sum() + wind_gen_series.sum()
        pv_gen = pv_gen_series.sum()
        wind_gen = wind_gen_series.sum()

        ratio_pv = np.divide(pv_gen, gen_total, out=np.zeros_like(pv_gen), where=gen_total > 0)
        ratio_wind = np.divide(wind_gen, gen_total, out=np.zeros_like(wind_gen), where=gen_total > 0)

        pv_used = pv_gen - (curtail * ratio_pv)
        wind_used = wind_gen - (curtail * ratio_wind)

        doc = (grid_buy * PRICE_GRID) + (pv_used * PRICE_PV) + (wind_used * PRICE_Wind)
        return {"doc": doc, "grid_buy": grid_buy, "curtail": curtail}

    # --- 储能模拟 ---
    soc = np.zeros(25)
    soc[0] = soc_min
    p_charge = np.zeros(24)
    p_discharge = np.zeros(24)
    grid_buy_new = np.zeros(24)
    curtail_new = np.zeros(24)

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

    total_grid_buy_new = grid_buy_new.sum()
    total_curtail_new = curtail_new.sum()
    total_pv_gen = pv_gen_series.sum()
    total_wind_gen = wind_gen_series.sum()
    total_gen = total_pv_gen + total_wind_gen
    ratio_pv = np.divide(total_pv_gen, total_gen, out=np.zeros_like(total_pv_gen), where=total_gen > 0)
    ratio_wind = np.divide(total_wind_gen, total_gen, out=np.zeros_like(total_wind_gen), where=total_gen > 0)
    total_pv_used_new = total_pv_gen - (total_curtail_new * ratio_pv)
    total_wind_used_new = total_wind_gen - (total_curtail_new * ratio_wind)
    doc = (total_grid_buy_new * PRICE_GRID) + (total_pv_used_new * PRICE_PV) + (total_wind_used_new * PRICE_Wind)

    return {
        "doc": doc,
        "grid_buy": total_grid_buy_new,
        "curtail": total_curtail_new
    }


def calculate_dic(p_cap, e_cap):
    """计算日均投资成本(DIC)"""
    return (p_cap * COST_P_ES + e_cap * COST_E_ES) / LIFESPAN_DAYS


def find_optimal_config(load_series, pv_gen_series, wind_gen_series, p_range, e_range):
    """
    为单个园区执行遍历搜索，找到最优 P, E 配置
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
    return optimal_config


# --- 5. 主程序 ---
if __name__ == "__main__":

    print("--- 正在执行 问题1(3) [独立运营最优解] ... ---")

    # 定义独立搜索范围
    P_range_ind = np.arange(0, 151, 10)  # 功率(kW): 从0到150, 步长10
    E_range_ind = np.arange(0, 301, 20)  # 容量(kWh): 从0到300, 步长20

    optimal_A = find_optimal_config(df['Load_A'], df['Gen_A'], zero_series, P_range_ind, E_range_ind)
    optimal_B = find_optimal_config(df['Load_B'], zero_series, df['Gen_B'], P_range_ind, E_range_ind)
    optimal_C = find_optimal_config(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], P_range_ind, E_range_ind)

    # (1) 计算“独立运营”的总成本
    tdc_independent_total = optimal_A['TDC'] + optimal_B['TDC'] + optimal_C['TDC']

    print("\n--- 正在执行 问题2(2) [联合运营最优解] ... ---")

    # 定义联合搜索范围
    P_range_joint = np.arange(0, 301, 20)  # 功率(kW): 从0到300, 步长20
    E_range_joint = np.arange(0, 601, 40)  # 容量(kWh): 从0到600, 步长40

    # (2) 计算“联合运营”的总成本
    optimal_joint = find_optimal_config(
        df['Load_Total'],
        df['Gen_PV_Total'],
        df['Gen_Wind_Total'],
        P_range_joint,
        E_range_joint
    )
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

    print("\n[导致经济收益改变的主要因素分析]")
    print("1. **时空互补性 (主要因素)**：联合运营允许三个园区的负荷和发电在空间上进行互补。")
    print("   例如，A园区的多余光伏可以供给B园区的负荷，避免了独立运营时“A弃光、B购电”的双重损失。")
    print("2. **规模经济性**：联合运营使用一个统一的大型储能系统。")
    print("   其总投资成本 (DIC) 可能低于三个园区各自配置储能的投资成本之和，实现了投资上的规模效益。")