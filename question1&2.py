import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 1. 导入图表库

# --- 1. 定义常量 ---
CAP_PV_A = 750.0  # A区光伏容量 (kW)
CAP_Wind_B = 1000.0  # B区风电容量 (kW)
CAP_PV_C = 600.0  # C区光伏容量 (kW)
CAP_Wind_C = 500.0  # C区风电容量 (kW)

PRICE_GRID = 1.0  # 网购电价 (元/kWh)
PRICE_PV = 0.4  # 光伏成本 (元/kWh)
PRICE_Wind = 0.5  # 风电成本 (元/kWh)

# --- (新增) 设置 Matplotlib 中文字体 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置失败 (可能未安装 'SimHei')，图表标签可能显示异常: {e}")
# --- 结束新增 ---


# --- 2. 加载数据 ---
try:
    df_load = pd.read_excel("附件1：各园区典型日负荷数据.xlsx")
    df_gen_pu = pd.read_excel("附件2：各园区典型日风光发电数据.xlsx", header=None, skiprows=3)
except FileNotFoundError:
    print("错误：未找到附件xlsx文件。请确保文件与脚本在同一目录。")
    exit()
except ImportError:
    print("错误：缺少 'openpyxl' 库。请运行 'pip install openpyxl' 来安装。")
    exit()

# --- 3. 准备一个统一的DataFrame ---
df = pd.DataFrame()
df['Load_A'] = df_load['园区A负荷(kW)']
df['Load_B'] = df_load['园区B负荷(kW)']
df['Load_C'] = df_load['园区C负荷(kW)']
df['PV_A_pu'] = pd.to_numeric(df_gen_pu[1])
df['Wind_B_pu'] = pd.to_numeric(df_gen_pu[2])
df['PV_C_pu'] = pd.to_numeric(df_gen_pu[3])
df['Wind_C_pu'] = pd.to_numeric(df_gen_pu[4])

# --- 4. 园区A: 逐时模拟计算 ---
df['Gen_A'] = df['PV_A_pu'] * CAP_PV_A
df['Net_Load_A'] = df['Load_A'] - df['Gen_A']
df['Grid_Buy_A'] = np.where(df['Net_Load_A'] > 0, df['Net_Load_A'], 0)
df['Curtail_A'] = np.where(df['Net_Load_A'] <= 0, -df['Net_Load_A'], 0)
df['PV_Used_A'] = np.where(df['Net_Load_A'] > 0, df['Gen_A'], df['Load_A'])

# --- 5. 园区A: 汇总结果 ---
total_load_A = df['Load_A'].sum()
total_grid_buy_A = df['Grid_Buy_A'].sum()
total_curtail_A = df['Curtail_A'].sum()
total_pv_used_A = df['PV_Used_A'].sum()
cost_A = (total_grid_buy_A * PRICE_GRID) + (total_pv_used_A * PRICE_PV)
avg_cost_A = cost_A / total_load_A

# --- 6. 园区B: 逐时模拟计算 ---
df['Gen_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Net_Load_B'] = df['Load_B'] - df['Gen_B']
df['Grid_Buy_B'] = np.where(df['Net_Load_B'] > 0, df['Net_Load_B'], 0)
df['Curtail_B'] = np.where(df['Net_Load_B'] <= 0, -df['Net_Load_B'], 0)
df['Wind_Used_B'] = np.where(df['Net_Load_B'] > 0, df['Gen_B'], df['Load_B'])

# --- 7. 园区B: 汇总结果 ---
total_load_B = df['Load_B'].sum()
total_grid_buy_B = df['Grid_Buy_B'].sum()
total_curtail_B = df['Curtail_B'].sum()
total_wind_used_B = df['Wind_Used_B'].sum()
cost_B = (total_grid_buy_B * PRICE_GRID) + (total_wind_used_B * PRICE_Wind)
avg_cost_B = cost_B / total_load_B

# --- 8. 园区C: 逐时模拟计算 ---
df['Gen_PV_C'] = pd.to_numeric(df['PV_C_pu']) * CAP_PV_C
df['Gen_Wind_C'] = pd.to_numeric(df['Wind_C_pu']) * CAP_Wind_C
df['Gen_C_Total'] = df['Gen_PV_C'] + df['Gen_Wind_C']
df['Net_Load_C'] = df['Load_C'] - df['Gen_C_Total']
df['Grid_Buy_C'] = np.where(df['Net_Load_C'] > 0, df['Net_Load_C'], 0)
df['Curtail_C'] = np.where(df['Net_Load_C'] <= 0, -df['Net_Load_C'], 0)
ratio_pv_C = np.divide(df['Gen_PV_C'], df['Gen_C_Total'], out=np.zeros_like(df['Gen_PV_C']),
                       where=df['Gen_C_Total'] > 0)
ratio_wind_C = np.divide(df['Gen_Wind_C'], df['Gen_C_Total'], out=np.zeros_like(df['Gen_Wind_C']),
                         where=df['Gen_C_Total'] > 0)
df['PV_Used_C'] = np.where(df['Net_Load_C'] > 0, df['Gen_PV_C'], df['Load_C'] * ratio_pv_C)
df['Wind_Used_C'] = np.where(df['Net_Load_C'] > 0, df['Gen_Wind_C'], df['Load_C'] * ratio_wind_C)

# --- 9. 园区C: 汇总结果 ---
total_load_C = df['Load_C'].sum()
total_grid_buy_C = df['Grid_Buy_C'].sum()
total_curtail_C = df['Curtail_C'].sum()
total_pv_used_C = df['PV_Used_C'].sum()
total_wind_used_C = df['Wind_Used_C'].sum()
cost_C = (total_grid_buy_C * PRICE_GRID) + (total_pv_used_C * PRICE_PV) + (total_wind_used_C * PRICE_Wind)
avg_cost_C = cost_C / total_load_C

# --- 10. 打印所有结果 ---
print("--- 问题1(1) 无储能经济性分析结果 ---")
print(f"\n--- 园区A ---")
print(f"总购电量: {total_grid_buy_A:,.2f} kWh")
print(f"总弃光量: {total_curtail_A:,.2f} kWh")
print(f"总供电成本: {cost_A:,.2f} 元")
print(f"单位平均成本: {avg_cost_A:.4f} 元/kWh")
# ... (园区B, C的打印)
print(f"\n--- 园区B ---")
print(f"总购电量: {total_grid_buy_B:,.2f} kWh")
print(f"总弃风量: {total_curtail_B:,.2f} kWh")
print(f"总供电成本: {cost_B:,.2f} 元")
print(f"单位平均成本: {avg_cost_B:.4f} 元/kWh")
print(f"\n--- 园区C ---")
print(f"总购电量: {total_grid_buy_C:,.2f} kWh")
print(f"总弃电量: {total_curtail_C:,.2f} kWh")
print(f"总供电成本: {cost_C:,.2f} 元")
print(f"单位平均成本: {avg_cost_C:.4f} 元/kWh")
print("\n关键因素分析：...")  # (省略)

# --- 11. 问题1(2): 50kW/100kWh 储能分析 ---
print("\n\n--- 问题1(2) 固定储能(50kW/100kWh)经济性分析 ---")

# --- 定义储能参数 ---
P_ES = 50.0
E_ES = 100.0
SOC_MIN = E_ES * 0.1
SOC_MAX = E_ES * 0.9
ETA = 0.95


# --- (!! 关键 Bug 修复 !!) ---
# 1. 增加了一个 `load_series` 参数
# 2. 修复了 "新单位成本" 的计算
def simulate_with_storage(net_load_series, pv_gen_series, wind_gen_series, load_series):
    soc = np.zeros(25);
    soc[0] = SOC_MIN
    p_charge = np.zeros(24);
    p_discharge = np.zeros(24)
    grid_buy_new = np.zeros(24);
    curtail_new = np.zeros(24)

    for t in range(24):
        net_load = net_load_series[t]
        if net_load > 0:
            max_can_discharge = min(P_ES, (soc[t] - SOC_MIN) * ETA)
            p_discharge[t] = min(max_can_discharge, net_load)
            grid_buy_new[t] = net_load - p_discharge[t]
        elif net_load < 0:
            surplus = -net_load
            max_can_charge = min(P_ES, (SOC_MAX - soc[t]) / ETA)
            p_charge[t] = min(max_can_charge, surplus)
            curtail_new[t] = surplus - p_charge[t]
        soc[t + 1] = soc[t] - (p_discharge[t] / ETA) + (p_charge[t] * ETA)

    total_grid_buy_new = grid_buy_new.sum()
    total_curtail_new = curtail_new.sum()
    total_pv_gen = pv_gen_series.sum()
    total_wind_gen = wind_gen_series.sum()
    total_gen = total_pv_gen + total_wind_gen
    ratio_pv = np.divide(total_pv_gen, total_gen, out=np.zeros_like(total_pv_gen), where=total_gen > 0)
    ratio_wind = np.divide(total_wind_gen, total_gen, out=np.zeros_like(total_wind_gen), where=total_gen > 0)
    total_pv_used_new = total_pv_gen - (total_curtail_new * ratio_pv)
    total_wind_used_new = total_wind_gen - (total_curtail_new * ratio_wind)
    cost_grid = total_grid_buy_new * PRICE_GRID
    cost_pv = total_pv_used_new * PRICE_PV
    cost_wind = total_wind_used_new * PRICE_Wind
    total_cost_new = cost_grid + cost_pv + cost_wind

    # 修复：
    total_load_kwh = load_series.sum()

    return {
        "新总购电量": total_grid_buy_new,
        "新总弃电量": total_curtail_new,
        "新总成本": total_cost_new,
        # 修复：除以各自园区的总负荷
        "新单位成本": total_cost_new / total_load_kwh if total_load_kwh > 0 else 0
    }


# --- (Bug 修复结束) ---


# --- 执行模拟 (已修改：传入了 load_series) ---
results_A_storage = simulate_with_storage(
    df['Net_Load_A'],
    df['Gen_A'],
    pd.Series(np.zeros(24)),
    df['Load_A']  # 传入A的负荷
)
results_B_storage = simulate_with_storage(
    df['Net_Load_B'],
    pd.Series(np.zeros(24)),
    df['Gen_B'],
    df['Load_B']  # 传入B的负荷
)
results_C_storage = simulate_with_storage(
    df['Net_Load_C'],
    df['Gen_PV_C'],
    df['Gen_Wind_C'],
    df['Load_C']  # 传入C的负荷
)


# --- 打印对比结果 ---
def print_comparison(park_name, old_cost, new_cost, old_grid, new_grid, old_curtail, new_curtail):
    print(f"\n--- {park_name} 对比 ---")
    print(f"总成本: {old_cost:,.2f} 元  ->  {new_cost:,.2f} 元")
    print(f"购电量: {old_grid:,.2f} kWh ->  {new_grid:,.2f} kWh")
    print(f"弃电量: {old_curtail:,.2f} kWh ->  {new_curtail:,.2f} kWh")
    if old_cost > new_cost:
        print(f"结论: 经济性改善，节省 {old_cost - new_cost:,.2f} 元。")
    else:
        print("结论: 经济性未改善。")


print_comparison("园区A", cost_A, results_A_storage['新总成本'], total_grid_buy_A, results_A_storage['新总购电量'],
                 total_curtail_A, results_A_storage['新总弃电量'])
print_comparison("园区B", cost_B, results_B_storage['新总成本'], total_grid_buy_B, results_B_storage['新总购电量'],
                 total_curtail_B, results_B_storage['新总弃电量'])
print_comparison("园区C", cost_C, results_C_storage['新总成本'], total_grid_buy_C, results_C_storage['新总购电量'],
                 total_curtail_C, results_C_storage['新总弃电量'])

print("\n原因分析：...")  # (省略)


# --- 12. (!! 新增模块 !!) 可视化图表 ---

def plot_comparison_charts(results_before, results_after):
    """
    生成并保存三个对比图表
    """
    parks = ['园区A', '园区B', '园区C']

    # 1. 提取数据
    costs_before = [results_before['A']['cost'], results_before['B']['cost'], results_before['C']['cost']]
    costs_after = [results_after['A']['cost'], results_after['B']['cost'], results_after['C']['cost']]

    grid_buy_before = [results_before['A']['grid_buy'], results_before['B']['grid_buy'],
                       results_before['C']['grid_buy']]
    grid_buy_after = [results_after['A']['grid_buy'], results_after['B']['grid_buy'], results_after['C']['grid_buy']]

    curtail_before = [results_before['A']['curtail'], results_before['B']['curtail'], results_before['C']['curtail']]
    curtail_after = [results_after['A']['curtail'], results_after['B']['curtail'], results_after['C']['curtail']]

    x = np.arange(len(parks))  # 标签位置
    width = 0.35  # 柱子宽度

    # --- 图 1: 总成本对比 ---
    try:
        fig1, ax1 = plt.subplots(layout='constrained')
        rects1 = ax1.bar(x - width / 2, costs_before, width, label='无储能', color='salmon')
        rects2 = ax1.bar(x + width / 2, costs_after, width, label='50/100 储能', color='dodgerblue')

        ax1.set_ylabel('总供电成本 (元)')
        ax1.set_title('各园区加装储能前后 总成本对比')
        ax1.set_xticks(x, parks)
        ax1.legend()
        ax1.bar_label(rects1, padding=3, fmt='%.0f')
        ax1.bar_label(rects2, padding=3, fmt='%.0f')

        plt.savefig("plot_1_cost_comparison.png")

        # --- 图 2: 总购电量对比 ---
        fig2, ax2 = plt.subplots(layout='constrained')
        rects3 = ax2.bar(x - width / 2, grid_buy_before, width, label='无储能', color='salmon')
        rects4 = ax2.bar(x + width / 2, grid_buy_after, width, label='50/100 储能', color='dodgerblue')

        ax2.set_ylabel('总购电量 (kWh)')
        ax2.set_title('各园区加装储能前后 总购电量对比')
        ax2.set_xticks(x, parks)
        ax2.legend()
        ax2.bar_label(rects3, padding=3, fmt='%.0f')
        ax2.bar_label(rects4, padding=3, fmt='%.0f')

        plt.savefig("plot_2_grid_buy_comparison.png")

        # --- 图 3: 总弃电量对比 ---
        fig3, ax3 = plt.subplots(layout='constrained')
        rects5 = ax3.bar(x - width / 2, curtail_before, width, label='无储能', color='salmon')
        rects6 = ax3.bar(x + width / 2, curtail_after, width, label='50/100 储能', color='dodgerblue')

        ax3.set_ylabel('总弃电量 (kWh)')
        ax3.set_title('各园区加装储能前后 总弃电量对比')
        ax3.set_xticks(x, parks)
        ax3.legend()
        ax3.bar_label(rects5, padding=3, fmt='%.0f')
        ax3.bar_label(rects6, padding=3, fmt='%.0f')

        plt.savefig("plot_3_curtailment_comparison.png")

        print("\n--- 可视化图表已生成 ---")
        print("已保存 'plot_1_cost_comparison.png'")
        print("已保存 'plot_2_grid_buy_comparison.png'")
        print("已保存 'plot_3_curtailment_comparison.png'")
        # plt.show() # 在脚本中，保存文件比显示更好

    except Exception as e:
        print(f"\n--- 可视化图表生成失败 ---")
        print(f"错误: {e}")
        print("请确保已安装 matplotlib (pip install matplotlib) 并且中文字体 'SimHei' 已安装。")


# --- (新增) 整理数据并调用绘图函数 ---
# 将所有结果打包进字典，方便绘图函数调用
results_before_all = {
    'A': {'cost': cost_A, 'grid_buy': total_grid_buy_A, 'curtail': total_curtail_A},
    'B': {'cost': cost_B, 'grid_buy': total_grid_buy_B, 'curtail': total_curtail_B},
    'C': {'cost': cost_C, 'grid_buy': total_grid_buy_C, 'curtail': total_curtail_C}
}

results_after_all = {
    'A': {'cost': results_A_storage['新总成本'], 'grid_buy': results_A_storage['新总购电量'],
          'curtail': results_A_storage['新总弃电量']},
    'B': {'cost': results_B_storage['新总成本'], 'grid_buy': results_B_storage['新总购电量'],
          'curtail': results_B_storage['新总弃电量']},
    'C': {'cost': results_C_storage['新总成本'], 'grid_buy': results_C_storage['新总购电量'],
          'curtail': results_C_storage['新总弃电量']}
}

# 调用绘图函数
plot_comparison_charts(results_before_all, results_after_all)
# --- (新增模块结束) ---