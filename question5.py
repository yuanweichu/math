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
    # 跳过前3行表头，从第4行(索引3)开始读，且没有表头(header=None)
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

# --- 3. 计算“联合园区”的 总负荷 和 总发电 ---
df['Gen_PV_A'] = df['PV_A_pu'] * CAP_PV_A
df['Gen_Wind_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Gen_PV_C'] = df['PV_C_pu'] * CAP_PV_C
df['Gen_Wind_C'] = df['Wind_C_pu'] * CAP_Wind_C

# (1) 计算联合总负荷
df['Load_Total'] = df['Load_A'] + df['Load_B'] + df['Load_C']
# (2) 计算联合总光伏发电
df['Gen_PV_Total'] = df['Gen_PV_A'] + df['Gen_PV_C']
# (3) 计算联合总风电发电
df['Gen_Wind_Total'] = df['Gen_Wind_B'] + df['Gen_Wind_C']
# (4) 计算联合总发电
df['Gen_Total'] = df['Gen_PV_Total'] + df['Gen_Wind_Total']


# --- 4. 定义核心函数 (已按 critique 修正) ---

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
    gen_total_series = pv_gen_series + wind_gen_series

    # --- 修正点 1: “无储能”分支的成本计算 ---
    # 如果p或e为0，则执行“无储能”的DOC计算 (采用逐时向量化计算)
    if e_es <= 0 or p_es <= 0:
        grid_buy = np.where(net_load_series > 0, net_load_series, 0)
        curtail = np.where(net_load_series <= 0, -net_load_series, 0)

        # 逐时计算可再生能源使用量
        ren_used_series = gen_total_series - curtail

        # 逐时计算风光比例
        ratio_pv = np.divide(pv_gen_series, gen_total_series, out=np.zeros_like(pv_gen_series),
                             where=gen_total_series > 0)
        ratio_wind = np.divide(wind_gen_series, gen_total_series, out=np.zeros_like(wind_gen_series),
                               where=gen_total_series > 0)

        # 逐时计算风光使用量
        pv_used = ren_used_series * ratio_pv
        wind_used = ren_used_series * ratio_wind

        # 计算总成本
        doc = (grid_buy.sum() * PRICE_GRID) + (pv_used.sum() * PRICE_PV) + (wind_used.sum() * PRICE_Wind)
        return {"doc": doc, "grid_buy": grid_buy.sum(), "curtail": curtail.sum()}
    # --- 修正点 1 结束 ---

    # --- 修正点 2: “有储能”分支的成本计算 ---
    # --- 储能模拟开始 ---
    soc = np.zeros(25)
    soc[0] = soc_min
    p_charge = np.zeros(24)
    p_discharge = np.zeros(24)
    grid_buy_new = np.zeros(24)
    curtail_new = np.zeros(24)

    # *** 新增：用于逐时成本计算 ***
    pv_used_new = np.zeros(24)
    wind_used_new = np.zeros(24)
    # *** 新增结束 ***

    for t in range(24):
        net_load = net_load_series[t]

        if net_load > 0:  # 缺电
            max_can_discharge = min(p_es, (soc[t] - soc_min) * eta)
            p_discharge[t] = min(max_can_discharge, net_load)
            grid_buy_new[t] = net_load - p_discharge[t]
        elif net_load < 0:  # 电多余
            surplus = -net_load
            max_can_charge = min(p_es, (soc_max - soc[t]) / eta)
            p_charge[t] = min(max_can_charge, surplus)
            curtail_new[t] = surplus - p_charge[t]

        soc[t + 1] = soc[t] - (p_discharge[t] / eta) + (p_charge[t] * eta)

        # --- *** 关键修正：逐时计算可再生能源使用量 *** ---
        pv_gen_t = pv_gen_series[t]
        wind_gen_t = wind_gen_series[t]
        gen_total_t = pv_gen_t + wind_gen_t

        # 该小时实际使用的可再生能源 = 该小时总发电 - 该小时弃电
        ren_used_t = gen_total_t - curtail_new[t]

        # 计算该小时的风光比例
        ratio_pv_t = np.divide(pv_gen_t, gen_total_t, out=np.zeros_like(pv_gen_t), where=gen_total_t > 0)
        ratio_wind_t = np.divide(wind_gen_t, gen_total_t, out=np.zeros_like(wind_gen_t), where=gen_total_t > 0)

        # 计算该小时的风光实际使用量
        pv_used_new[t] = ren_used_t * ratio_pv_t
        wind_used_new[t] = ren_used_t * ratio_wind_t
        # --- *** 修正结束 *** ---

    # --- 储能模拟结束 ---

    # --- 循环结束，计算新的经济账 (基于逐时计算结果求和) ---
    total_grid_buy_new = grid_buy_new.sum()
    total_curtail_new = curtail_new.sum()

    # *** 修正：直接对逐时使用量求和 ***
    total_pv_used_new = pv_used_new.sum()
    total_wind_used_new = wind_used_new.sum()

    # *** 修正：DOC 基于新的使用量计算 ***
    doc = (total_grid_buy_new * PRICE_GRID) + (total_pv_used_new * PRICE_PV) + (total_wind_used_new * PRICE_Wind)

    return {
        "doc": doc,
        "grid_buy": total_grid_buy_new,
        "curtail": total_curtail_new
    }
    # --- 修正点 2 结束 ---


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


# --- 5. 主程序 (无需修改) ---
if __name__ == "__main__":

    print("--- 问题2(1) 联合园区 (无储能) 经济性分析 ---")

    # 运行“无储能”分析，作为基准
    baseline_results = analyze_doc_with_storage(
        df['Load_Total'], df['Gen_PV_Total'], df['Gen_Wind_Total'], 0, 0
    )
    baseline_tdc = baseline_results['doc']  # 此时 DIC = 0

    print(f"联合总购电量:       {baseline_results['grid_buy']:,.2f} kWh")
    print(f"联合总弃风弃光电量: {baseline_results['curtail']:,.2f} kWh")
    print(f"联合总供电成本:     {baseline_tdc:,.2f} 元")
    print(f"联合单位平均成本:   {baseline_tdc / df['Load_Total'].sum():,.4f} 元/kWh")

    print("\n\n--- 问题2(2) 联合园区 (最优储能) 配置分析 ---")
    start_time = time.time()

    # --- 搜索范围定义 ---
    # 联合园区的负荷和波动更大，我们可能需要更大的储能，因此扩大搜索范围
    P_range = np.arange(0, 301, 20)  # 功率(kW): 从0到300, 步长20
    E_range = np.arange(0, 601, 40)  # 容量(kWh): 从0到600, 步长40

    print(f"正在执行遍历搜索 (P: {len(P_range)} 步, E: {len(E_range)} 步)...")

    # --- G核心：对“联合数据”执行一次优化 ---
    optimal_combined = find_optimal_config(
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