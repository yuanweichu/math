import pandas as pd
import numpy as np

# --- 1. 定义常量 ---
CAP_PV_A = 750.0      # A区光伏容量 (kW)
CAP_Wind_B = 1000.0   # B区风电容量 (kW)
CAP_PV_C = 600.0      # C区光伏容量 (kW)
CAP_Wind_C = 500.0    # C区风电容量 (kW)

PRICE_GRID = 1.0    # 网购电价 (元/kWh)
PRICE_PV = 0.4      # 光伏成本 (元/kWh)
PRICE_Wind = 0.5    # 风电成本 (元/kWh)

# --- 2. 加载数据 ---

try:
    df_load = pd.read_excel("附件1：各园区典型日负荷数据.xlsx")
    # 1. 跳过前3行，从第4行（索引3）开始读取数据
    # 2. 告诉pandas这一行没有表头 (header=None)
    df_gen_pu = pd.read_excel("附件2：各园区典型日风光发电数据.xlsx", header=None, skiprows=3)

except FileNotFoundError:
    print("错误：未找到附件xlsx文件。请确保文件与脚本在同一目录。")
    exit()
# --- 3. 准备一个统一的DataFrame ---
df = pd.DataFrame()
df['Load_A'] = df_load['园区A负荷(kW)']
df['Load_B'] = df_load['园区B负荷(kW)']
df['Load_C'] = df_load['园区C负荷(kW)']
# 3.1 因为我们跳过了表头，所以现在按列的“位置索引”来提取数据
# （第0列是时间, 第1列是A, 第2列是B, 第3列是C的光伏, 第4列是C的风电）
df['PV_A_pu'] = df_gen_pu[1]
df['Wind_B_pu'] = df_gen_pu[2]
df['PV_C_pu'] = df_gen_pu[3]
df['Wind_C_pu'] = df_gen_pu[4]

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

# 计算C区使用量 (按比例)
ratio_pv_C = np.divide(df['Gen_PV_C'], df['Gen_C_Total'], out=np.zeros_like(df['Gen_PV_C']), where=df['Gen_C_Total'] > 0)
ratio_wind_C = np.divide(df['Gen_Wind_C'], df['Gen_C_Total'], out=np.zeros_like(df['Gen_Wind_C']), where=df['Gen_C_Total'] > 0)
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