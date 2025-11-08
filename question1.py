import pandas as pd
import numpy as np

#1. 定义常量
CAP_PV_A = 750.0      # A区光伏容量 (kW)
CAP_Wind_B = 1000.0   # B区风电容量 (kW)
CAP_PV_C = 600.0      # C区光伏容量 (kW)
CAP_Wind_C = 500.0    # C区风电容量 (kW)

PRICE_GRID = 1.0    # 网购电价 (元/kWh)
PRICE_PV = 0.4      # 光伏成本 (元/kWh)
PRICE_Wind = 0.5    # 风电成本 (元/kWh)

#2. 加载数据

try:
    df_load = pd.read_excel("附件1：各园区典型日负荷数据.xlsx")
    # 1. 跳过前3行，从第4行（索引3）开始读取数据
    df_gen_pu = pd.read_excel("附件2：各园区典型日风光发电数据.xlsx", header=None, skiprows=3)

except FileNotFoundError:
    print("错误：未找到附件xlsx文件。请确保文件与脚本在同一目录。")
    exit()
#3. 准备一个统一的DataFrame
df = pd.DataFrame()
df['Load_A'] = df_load['园区A负荷(kW)']
df['Load_B'] = df_load['园区B负荷(kW)']
df['Load_C'] = df_load['园区C负荷(kW)']
# （第0列是时间, 第1列是A, 第2列是B, 第3列是C的光伏, 第4列是C的风电）
df['PV_A_pu'] = df_gen_pu[1]
df['Wind_B_pu'] = df_gen_pu[2]
df['PV_C_pu'] = df_gen_pu[3]
df['Wind_C_pu'] = df_gen_pu[4]

#4. 园区A: 逐时模拟计算
df['Gen_A'] = df['PV_A_pu'] * CAP_PV_A
df['Net_Load_A'] = df['Load_A'] - df['Gen_A']
df['Grid_Buy_A'] = np.where(df['Net_Load_A'] > 0, df['Net_Load_A'], 0)
df['Curtail_A'] = np.where(df['Net_Load_A'] <= 0, -df['Net_Load_A'], 0)
df['PV_Used_A'] = np.where(df['Net_Load_A'] > 0, df['Gen_A'], df['Load_A'])

#5. 园区A: 汇总结果
total_load_A = df['Load_A'].sum()
total_grid_buy_A = df['Grid_Buy_A'].sum()
total_curtail_A = df['Curtail_A'].sum()
total_pv_used_A = df['PV_Used_A'].sum()

cost_A = (total_grid_buy_A * PRICE_GRID) + (total_pv_used_A * PRICE_PV)
avg_cost_A = cost_A / total_load_A

#6. 园区B: 逐时模拟计算
df['Gen_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Net_Load_B'] = df['Load_B'] - df['Gen_B']
df['Grid_Buy_B'] = np.where(df['Net_Load_B'] > 0, df['Net_Load_B'], 0)
df['Curtail_B'] = np.where(df['Net_Load_B'] <= 0, -df['Net_Load_B'], 0)
df['Wind_Used_B'] = np.where(df['Net_Load_B'] > 0, df['Gen_B'], df['Load_B'])

#7. 园区B: 汇总结果
total_load_B = df['Load_B'].sum()
total_grid_buy_B = df['Grid_Buy_B'].sum()
total_curtail_B = df['Curtail_B'].sum()
total_wind_used_B = df['Wind_Used_B'].sum()

cost_B = (total_grid_buy_B * PRICE_GRID) + (total_wind_used_B * PRICE_Wind)
avg_cost_B = cost_B / total_load_B

#8.园区C: 逐时模拟计算
df['Gen_PV_C'] = pd.to_numeric(df['PV_C_pu']) * CAP_PV_C
df['Gen_Wind_C'] = pd.to_numeric(df['Wind_C_pu']) * CAP_Wind_C
df['Gen_C_Total'] = df['Gen_PV_C'] + df['Gen_Wind_C']
df['Net_Load_C'] = df['Load_C'] - df['Gen_C_Total']
df['Grid_Buy_C'] = np.where(df['Net_Load_C'] > 0, df['Net_Load_C'], 0)
df['Curtail_C'] = np.where(df['Net_Load_C'] <= 0, -df['Net_Load_C'], 0)

# 计算C区使用量 (按比例)
ratio_pv_C = np.divide(df['Gen_PV_C'], df['Gen_C_Total'], out=np.zeros_like(df['Gen_PV_C']), where=df['Gen_C_Total'] > 0)
ratio_wind_C = np.divide(df['Gen_Wind_C'], df['Gen_C_Total'], out=np.zeros_like(df['Gen_Wind_C']), where=df['Gen_C_Total'] > 0)
df['PV_Used_C'] = np.where(df['Net_Load_C'] > 0, df['Gen_PV_C'], df['Load_C'] * ratio_pv_C)
df['Wind_Used_C'] = np.where(df['Net_Load_C'] > 0, df['Gen_Wind_C'], df['Load_C'] * ratio_wind_C)

#9. 园区C: 汇总结果
total_load_C = df['Load_C'].sum()
total_grid_buy_C = df['Grid_Buy_C'].sum()
total_curtail_C = df['Curtail_C'].sum()
total_pv_used_C = df['PV_Used_C'].sum()
total_wind_used_C = df['Wind_Used_C'].sum()

cost_C = (total_grid_buy_C * PRICE_GRID) + (total_pv_used_C * PRICE_PV) + (total_wind_used_C * PRICE_Wind)
avg_cost_C = cost_C / total_load_C

#10. 打印所有结果
print("--- 问题1(1) 无储能经济性分析结果 ---")

print(f"\n--- 园区A ---")
print(f"总购电量: {total_grid_buy_A:,.2f} kWh")
print(f"总弃光量: {total_curtail_A:,.2f} kWh")
print(f"总供电成本: {cost_A:,.2f} 元")
print(f"单位平均成本: {avg_cost_A:.4f} 元/kWh")

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

print("\n关键因素分析：")
print("经济性差的主要原因是“时序不匹配”：")
print("1. 发电高峰 (如中午光伏) 与用电高峰不一致，导致发电时用不完，产生“弃电”。")
print("2. 用电高峰 (如傍晚) 时发电量不足，必须从主网“高价购电”。")

# --- 11. 问题1(2): 50kW/100kWh 储能分析 ---

print("\n\n--- 问题1(2) 固定储能(50kW/100kWh)经济性分析 ---")

# --- 定义储能参数 ---
P_ES = 50.0  # 储能功率 (kW)
E_ES = 100.0  # 储能容量 (kWh)
SOC_MIN = E_ES * 0.1  # 最小荷电量 (kWh)
SOC_MAX = E_ES * 0.9  # 最大荷电量 (kWh)
ETA = 0.95  # 充放电效率


# 我们需要一个函数来执行这个逐时模拟
def simulate_with_storage(net_load_series, pv_gen_series, wind_gen_series):
    # 初始化24小时的储能状态
    soc = np.zeros(25)  # 25个点，soc[0]是初始值, soc[24]是终值
    soc[0] = SOC_MIN  # 假设一天开始时储能为空

    p_charge = np.zeros(24)  # 实际充电
    p_discharge = np.zeros(24)  # 实际放电
    grid_buy_new = np.zeros(24)  # 新的购电
    curtail_new = np.zeros(24)  # 新的弃电

    for t in range(24):
        net_load = net_load_series[t]

        # 1. IF net_load > 0 (缺电，尝试放电)
        if net_load > 0:
            # 储能最多能放多少电 (受限于功率、SOC下限、效率)
            max_can_discharge = min(P_ES, (soc[t] - SOC_MIN) * ETA)
            # 实际放电量 (不能超过缺口)
            p_discharge[t] = min(max_can_discharge, net_load)

            # 放电后，仍然不足的电量才从主网购买
            grid_buy_new[t] = net_load - p_discharge[t]

        # 2. IF net_load < 0 (电多余，尝试充电)
        elif net_load < 0:
            surplus = -net_load  # 多余的电量

            # 储能最多能充多少电 (受限于功率、SOC上限、效率)
            max_can_charge = min(P_ES, (SOC_MAX - soc[t]) / ETA)
            # 实际充电量 (不能超过多余的电量)
            p_charge[t] = min(max_can_charge, surplus)

            # 充完电后，仍然多余的电量才被弃掉
            curtail_new[t] = surplus - p_charge[t]

        # 3. 更新下一小时的SOC
        # (注意效率：放电是SOC减少量 / eta, 充电是SOC增加量 * eta)
        soc[t + 1] = soc[t] - (p_discharge[t] / ETA) + (p_charge[t] * ETA)

    # --- 循环结束，计算新的经济账 ---

    # 3.1 计算新的总购电量和总弃电量
    total_grid_buy_new = grid_buy_new.sum()
    total_curtail_new = curtail_new.sum()

    # 3.2 计算新的可再生能源使用量
    # (总发电量 - 总弃电量 = 总使用量)
    total_pv_gen = pv_gen_series.sum()
    total_wind_gen = wind_gen_series.sum()
    total_gen = total_pv_gen + total_wind_gen

    # 弃电也需要按比例分摊
    ratio_pv = np.divide(total_pv_gen, total_gen, out=np.zeros_like(total_pv_gen), where=total_gen > 0)
    ratio_wind = np.divide(total_wind_gen, total_gen, out=np.zeros_like(total_wind_gen), where=total_gen > 0)

    total_pv_used_new = total_pv_gen - (total_curtail_new * ratio_pv)
    total_wind_used_new = total_wind_gen - (total_curtail_new * ratio_wind)

    # 3.3 计算新的总成本
    cost_grid = total_grid_buy_new * PRICE_GRID
    cost_pv = total_pv_used_new * PRICE_PV
    cost_wind = total_wind_used_new * PRICE_Wind
    total_cost_new = cost_grid + cost_pv + cost_wind

    return {
        "新总购电量": total_grid_buy_new,
        "新总弃电量": total_curtail_new,
        "新总成本": total_cost_new,
        "新单位成本": total_cost_new / df['Load_A'].sum()  # 假设总负荷不变
    }


# --- 执行模拟 ---

# 准备 园区A 的数据
results_A_storage = simulate_with_storage(
    df['Net_Load_A'],
    df['Gen_A'],
    pd.Series(np.zeros(24))  # 园区A没有风电
)

# 准备 园区B 的数据
results_B_storage = simulate_with_storage(
    df['Net_Load_B'],
    pd.Series(np.zeros(24)),  # 园区B没有光伏
    df['Gen_B']
)

# 准备 园区C 的数据
results_C_storage = simulate_with_storage(
    df['Net_Load_C'],
    df['Gen_PV_C'],
    df['Gen_Wind_C']
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


# (从第1关的结果中获取旧数据)
print_comparison("园区A", cost_A, results_A_storage['新总成本'], total_grid_buy_A, results_A_storage['新总购电量'],
                 total_curtail_A, results_A_storage['新总弃电量'])
print_comparison("园区B", cost_B, results_B_storage['新总成本'], total_grid_buy_B, results_B_storage['新总购电量'],
                 total_curtail_B, results_B_storage['新总弃电量'])
print_comparison("园区C", cost_C, results_C_storage['新总成本'], total_grid_buy_C, results_C_storage['新总购电量'],
                 total_curtail_C, results_C_storage['新总弃电量'])

print("\n原因分析：")
print("经济性改善的主要原因是：储能系统执行了“低储高放”的套利策略。")
print("1. 充电: 在发电多余、本应“弃电”(0成本)的时段，储能将这些免费电力储存起来。")
print("2. 放电: 在电力不足、本应“高价购电”(1.0元/kWh)的时段，储能释放储存的电力，替代了这部分高额成本。")