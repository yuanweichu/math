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
    print("请确保 .xlsx 文件存在，并且 'openpyxl' 库已安装 (pip install openpyxl)")
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

# --- 3. 计算基础数据 (净负荷) ---
df['Gen_A'] = df['PV_A_pu'] * CAP_PV_A
df['Net_Load_A'] = df['Load_A'] - df['Gen_A']

df['Gen_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Net_Load_B'] = df['Load_B'] - df['Gen_B']

df['Gen_PV_C'] = df['PV_C_pu'] * CAP_PV_C
df['Gen_Wind_C'] = df['Wind_C_pu'] * CAP_Wind_C
df['Gen_C_Total'] = df['Gen_PV_C'] + df['Gen_Wind_C']
df['Net_Load_C'] = df['Load_C'] - df['Gen_C_Total']

zero_series = pd.Series(np.zeros(24))


# --- 4. 关键函数：储能模拟 (已修复) ---

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


# --- 5. 新增：可复用的“遍历搜索”函数 (已修改) ---
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

            # --- 这是本次修改的核心 ---
            # 将 'grid_buy' 和 'curtail' 也保存到结果中
            results.append({
                'P_cap': p, 'E_cap': e, 'TDC': tdc, 'DOC': doc, 'DIC': dic,
                'Grid_Buy': sim_result['grid_buy'],
                'Curtail': sim_result['curtail']
            })
            # --- 修改结束 ---

    df_results = pd.DataFrame(results)
    # 找到TDC最小的行
    optimal_config = df_results.loc[df_results['TDC'].idxmin()]
    return optimal_config


# --- 6. 主程序：(已修改) ---

if __name__ == "__main__":

    print("--- 问题1(3) 最优储能配置分析 ---")
    total_start_time = time.time()

    # --- 搜索范围定义 ---
    P_range = np.arange(0, 151, 10)  # 功率(kW): 从0到150, 步长10
    E_range = np.arange(0, 301, 20)  # 容量(kWh): 从0到300, 步长20

    # ----------------------------------------------------
    # (1) 论证：50kW/100kWh 是否最优？
    # ----------------------------------------------------
    print("正在计算 50kW/100kWh 方案的TDC...")
    dic_50 = calculate_dic(50, 100)  # 50/100的投资成本是固定的

    doc_A_50 = analyze_doc_with_storage(df['Load_A'], df['Gen_A'], zero_series, 50, 100)['doc']
    tdc_A_50 = doc_A_50 + dic_50
    print(f"园区A (50/100) TDC: {tdc_A_50:,.2f} 元 (DOC: {doc_A_50:,.2f} + DIC: {dic_50:,.2f})")

    doc_B_50 = analyze_doc_with_storage(df['Load_B'], zero_series, df['Gen_B'], 50, 100)['doc']
    tdc_B_50 = doc_B_50 + dic_50
    print(f"园区B (50/100) TDC: {tdc_B_50:,.2f} 元 (DOC: {doc_B_50:,.2f} + DIC: {dic_50:,.2f})")

    doc_C_50 = analyze_doc_with_storage(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], 50, 100)['doc']
    tdc_C_50 = doc_C_50 + dic_50
    print(f"园区C (50/100) TDC: {tdc_C_50:,.2f} 元 (DOC: {doc_C_50:,.2f} + DIC: {dic_50:,.2f})")

    # ----------------------------------------------------
    # (2) 寻找最优解 (为A, B, C分别执行)
    # ----------------------------------------------------
    print(f"\n正在执行遍历搜索 (P: {len(P_range)} 步, E: {len(E_range)} 步)...")

    print("\n--- 正在优化 园区A ---")
    start_A = time.time()
    optimal_A = find_optimal_config(df['Load_A'], df['Gen_A'], zero_series, P_range, E_range)
    print(f"搜索完成！用时 {time.time() - start_A:.2f} 秒")

    print("\n--- 正在优化 园区B ---")
    start_B = time.time()
    optimal_B = find_optimal_config(df['Load_B'], zero_series, df['Gen_B'], P_range, E_range)
    print(f"搜索完成！用时 {time.time() - start_B:.2f} 秒")

    print("\n--- G正在优化 园区C ---")
    start_C = time.time()
    optimal_C = find_optimal_config(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], P_range, E_range)
    print(f"搜索完成！用时 {time.time() - start_C:.2f} 秒")

    # ----------------------------------------------------
    # (3) 打印最终的优化结果报告 (已修改)
    # ----------------------------------------------------
    print("\n\n" + "=" * 30)
    print("--- 最终优化结果报告 ---")
    print("=" * 30)

    # --- 打印园区A的结果 (已修改) ---
    print(f"\n[园区A 最优配置方案]")
    print(f"最优功率: {optimal_A['P_cap']} kW, 最优容量: {optimal_A['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_A['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_A['DOC']:,.2f} 元, DIC: {optimal_A['DIC']:,.2f} 元)")
    # --- 新增行 ---
    print(f" (最优方案详情: 购电 {optimal_A['Grid_Buy']:,.2f} kWh, 弃电 {optimal_A['Curtail']:,.2f} kWh)")

    print(f"[论证 50/100 方案]: {tdc_A_50:,.2f} 元 > 最优方案 {optimal_A['TDC']:,.2f} 元。")
    if tdc_A_50 > optimal_A['TDC']:
        print("结论：50/100 方案不是最优方案。")
    else:
        print("结论：50/100 方案在搜索范围内是最优方案。")

    # --- 打印园区B的结果 (已修改) ---
    print(f"\n[园区B 最优配置方案]")
    print(f"最优功率: {optimal_B['P_cap']} kW, 最优容量: {optimal_B['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_B['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_B['DOC']:,.2f} 元, DIC: {optimal_B['DIC']:,.2f} 元)")
    # --- 新增行 ---
    print(f" (最优方案详情: 购电 {optimal_B['Grid_Buy']:,.2f} kWh, 弃电 {optimal_B['Curtail']:,.2f} kWh)")

    print(f"[论证 50/100 方案]: {tdc_B_50:,.2f} 元 > 最优方案 {optimal_B['TDC']:,.2f} 元。")
    if tdc_B_50 > optimal_B['TDC']:
        print("结论：50/100 方案不是最优方案。")
    else:
        print("结论：50/100 方案在搜索范围内是最优方案。")

    # --- 打印园区C的结果 (已修改) ---
    print(f"\n[园区C 最优配置方案]")
    print(f"最优功率: {optimal_C['P_cap']} kW, 最优容量: {optimal_C['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_C['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_C['DOC']:,.2f} 元, DIC: {optimal_C['DIC']:,.2f} 元)")
    # --- 新增行 ---
    print(f" (最优方案详情: 购电 {optimal_C['Grid_Buy']:,.2f} kWh, 弃电 {optimal_C['Curtail']:,.2f} kWh)")

    print(f"[论证 50/100 方案]: {tdc_C_50:,.2f} 元 > 最优方案 {optimal_C['TDC']:,.2f} 元。")
    if tdc_C_50 > optimal_C['TDC']:
        print("结论：50/100 方案不是最优方案。")
    else:
        print("结论：50/100 方案在搜索范围内是最优方案。")

    print(f"\n--- 总用时: {time.time() - total_start_time:.2f} 秒 ---")