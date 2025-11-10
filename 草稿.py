import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt  # 1. 导入图表库

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

# --- 设置 Matplotlib 中文字体 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置失败 (可能未安装 'SimHei')，图表标签可能显示异常: {e}")

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


# --- 4. 关键函数：(已修正 UnboundLocalError) ---
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
    net_load_series = load_series - gen_total_series  # (已移到 if 之外)

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


def calculate_dic(p_cap, e_cap):
    """计算日均投资成本(DIC)"""
    return (p_cap * COST_P_ES + e_cap * COST_E_ES) / LIFESPAN_DAYS


# --- 5. 遍历搜索函数 ---
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


# --- 6. 主程序 ---
if __name__ == "__main__":

    print("--- 问题1(3) 最优储能配置分析 ---")
    total_start_time = time.time()
    P_range = np.arange(0, 151, 10)
    E_range = np.arange(0, 301, 20)

    # (1) 计算 (P=0, E=0) 基准方案
    print("\n--- 正在计算 (P=0, E=0) 基准方案的TDC... ---")
    baseline_A_result = analyze_doc_with_storage(df['Load_A'], df['Gen_A'], zero_series, 0, 0)
    tdc_A_baseline = baseline_A_result['doc']
    baseline_B_result = analyze_doc_with_storage(df['Load_B'], zero_series, df['Gen_B'], 0, 0)
    tdc_B_baseline = baseline_B_result['doc']
    baseline_C_result = analyze_doc_with_storage(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], 0, 0)
    tdc_C_baseline = baseline_C_result['doc']

    # (2) 计算 (50, 100) 固定方案
    print("正在计算 50kW/100kWh 方案的TDC...")
    dic_50 = calculate_dic(50, 100)
    doc_A_50 = analyze_doc_with_storage(df['Load_A'], df['Gen_A'], zero_series, 50, 100)['doc']
    tdc_A_50 = doc_A_50 + dic_50
    doc_B_50 = analyze_doc_with_storage(df['Load_B'], zero_series, df['Gen_B'], 50, 100)['doc']
    tdc_B_50 = doc_B_50 + dic_50
    doc_C_50 = analyze_doc_with_storage(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], 50, 100)['doc']
    tdc_C_50 = doc_C_50 + dic_50
    print(f"园区A (50/100) TDC: {tdc_A_50:,.2f} 元")
    print(f"园区B (50/100) TDC: {tdc_B_50:,.2f} 元")
    print(f"园区C (50/100) TDC: {tdc_C_50:,.2f} 元")

    # (3) 寻找最优解
    print(f"\n正在执行遍历搜索 (P: {len(P_range)} 步, E: {len(E_range)} 步)...")
    print("\n--- 正在优化 园区A ---")
    start_A = time.time()
    optimal_A = find_optimal_config(df['Load_A'], df['Gen_A'], zero_series, P_range, E_range)
    print(f"搜索完成！用时 {time.time() - start_A:.2f} 秒")
    print("\n--- 正在优化 园区B ---")
    start_B = time.time()
    optimal_B = find_optimal_config(df['Load_B'], zero_series, df['Gen_B'], P_range, E_range)
    print(f"搜索完成！用时 {time.time() - start_B:.2f} 秒")
    print("\n--- 正在优化 园区C ---")
    start_C = time.time()
    optimal_C = find_optimal_config(df['Load_C'], df['Gen_PV_C'], df['Gen_Wind_C'], P_range, E_range)
    print(f"搜索完成！用时 {time.time() - start_C:.2f} 秒")

    # (4) 打印最终的优化结果报告
    print("\n\n" + "=" * 30)
    print("--- 最终优化结果报告 ---")
    print("=" * 30)
    # (省略... 打印逻辑与之前相同)
    print(f"\n[园区A 最优配置方案]")
    print(f"最优功率: {optimal_A['P_cap']} kW, 最优容量: {optimal_A['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_A['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_A['DOC']:,.2f} 元, DIC: {optimal_A['DIC']:,.2f} 元)")
    print(f" (最优方案详情: 购电 {optimal_A['Grid_Buy']:,.2f} kWh, 弃电 {optimal_A['Curtail']:,.2f} kWh)")
    print(f"[论证 50/100 方案]: {tdc_A_50:,.2f} 元 >= 最优方案 {optimal_A['TDC']:,.2f} 元。")  # (修正了比较符)
    if tdc_A_50 >= optimal_A['TDC']:
        print("结论：50/100 方案不是最优方案 (或与最优方案相同)。")
    else:
        print("结论：50/100 方案在搜索范围内是最优方案。")
    print(f"\n[园区B 最优配置方案]")
    print(f"最优功率: {optimal_B['P_cap']} kW, 最优容量: {optimal_B['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_B['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_B['DOC']:,.2f} 元, DIC: {optimal_B['DIC']:,.2f} 元)")
    print(f" (最优方案详情: 购电 {optimal_B['Grid_Buy']:,.2f} kWh, 弃电 {optimal_B['Curtail']:,.2f} kWh)")
    print(f"[论证 50/100 方案]: {tdc_B_50:,.2f} 元 >= 最优方案 {optimal_B['TDC']:,.2f} 元。")
    if tdc_B_50 >= optimal_B['TDC']:
        print("结论：50/100 方案不是最优方案 (或与最优方案相同)。")
    else:
        print("结论：50/100 方案在搜索范围内是最优方案。")
    print(f"\n[园区C 最优配置方案]")
    print(f"最优功率: {optimal_C['P_cap']} kW, 最优容量: {optimal_C['E_cap']} kWh")
    print(f"最低总成本 TDC: {optimal_C['TDC']:,.2f} 元")
    print(f" (其中 DOC: {optimal_C['DOC']:,.2f} 元, DIC: {optimal_C['DIC']:,.2f} 元)")
    print(f" (最优方案详情: 购电 {optimal_C['Grid_Buy']:,.2f} kWh, 弃电 {optimal_C['Curtail']:,.2f} kWh)")
    print(f"[论证 50/100 方案]: {tdc_C_50:,.2f} 元 >= 最优方案 {optimal_C['TDC']:,.2f} 元。")
    if tdc_C_50 >= optimal_C['TDC']:
        print("结论：50/100 方案不是最优方案 (或与最优方案相同)。")
    else:
        print("结论：50/100 方案在搜索范围内是最优方案。")
    print(f"\n--- 总用时: {time.time() - total_start_time:.2f} 秒 ---")


    # ----------------------------------------------------
    # (5) (!! 已修改 !!) 可视化图表
    # ----------------------------------------------------
    def plot_optimal_comparison(baseline_costs, fixed_costs, optimal_costs):
        parks = ['园区A', '园区B', '园区C']
        data = {
            '基准 (无储能)': baseline_costs,
            '50/100 方案': fixed_costs,
            '最优储能方案': optimal_costs
        }

        x = np.arange(len(parks))
        width = 0.25
        multiplier = 0

        try:
            fig, ax = plt.subplots(layout='constrained')
            colors = ['salmon', 'cornflowerblue', 'forestgreen']
            i = 0

            for scenario, costs in data.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, costs, width, label=scenario, color=colors[i])
                ax.bar_label(rects, padding=3, fmt='%.0f', rotation=90, fontsize=8)
                multiplier += 1
                i += 1

            ax.set_ylabel('总成本 (TDC) (元)')
            ax.set_title('问题1(3): 各园区三种储能方案的总成本对比')
            ax.set_xticks(x + width, parks)
            ax.legend(loc='upper right', ncols=1)  # 修改图例位置

            # --- *** 关键修正 *** ---
            # 动态计算Y轴范围，实现“放大”
            all_costs = baseline_costs + fixed_costs + optimal_costs
            min_val = min(all_costs)
            max_val = max(all_costs)
            padding = (max_val - min_val) * 0.1  # 10% 的上下留白

            # 设置Y轴的下限和上限
            ax.set_ylim(bottom=min_val - padding, top=max_val + padding)
            # --- *** 修正结束 *** ---

            plt.savefig("plot_Q1_3_optimal_comparison.png")
            print("\n--- 可视化图表已生成 (已优化Y轴) ---")
            print("已保存 'plot_Q1_3_optimal_comparison.png'")

        except Exception as e:
            print(f"\n--- 可视化图表生成失败 ---")
            print(f"错误: {e}")


    # 整理数据并调用
    baseline_tdcs = [tdc_A_baseline, tdc_B_baseline, tdc_C_baseline]
    fixed_tdcs = [tdc_A_50, tdc_B_50, tdc_C_50]
    optimal_tdcs = [optimal_A['TDC'], optimal_B['TDC'], optimal_C['TDC']]

    plot_optimal_comparison(baseline_tdcs, fixed_tdcs, optimal_tdcs)