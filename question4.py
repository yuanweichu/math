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

# --- 3. 计算“联合园区”的 总负荷 和 总发电 ---

# (复用问题1的代码，计算各园区的发电)
df['Gen_PV_A'] = df['PV_A_pu'] * CAP_PV_A
df['Gen_Wind_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Gen_PV_C'] = df['PV_C_pu'] * CAP_PV_C
df['Gen_Wind_C'] = df['Wind_C_pu'] * CAP_Wind_C

# --- 这部分是问题2(1)的核心 ---
# (1) 计算联合总负荷
df['Load_Total'] = df['Load_A'] + df['Load_B'] + df['Load_C']

# (2) 计算联合总光伏发电
df['Gen_PV_Total'] = df['Gen_PV_A'] + df['Gen_PV_C']

# (3) 计算联合总风电发电
df['Gen_Wind_Total'] = df['Gen_Wind_B'] + df['Gen_Wind_C']

# (4) 计算联合总发电
df['Gen_Total'] = df['Gen_PV_Total'] + df['Gen_Wind_Total']
# --- 核心结束 ---


# --- 4. 联合园区 “无储能” 经济性分析 (同问题1(1)的逻辑) ---

# (1) 计算联合园区的净负荷
df['Net_Load_Total'] = df['Load_Total'] - df['Gen_Total']

# (2) 计算联合园区的购电量 和 弃电量
df['Grid_Buy_Total'] = np.where(df['Net_Load_Total'] > 0, df['Net_Load_Total'], 0)
df['Curtail_Total'] = np.where(df['Net_Load_Total'] <= 0, -df['Net_Load_Total'], 0)

# (3) 计算联合园区的实际使用量 (修正了图4的缺陷)
# 弃电量需要按风光比例分摊
total_gen = df['Gen_Total'].sum()
total_pv_gen = df['Gen_PV_Total'].sum()
total_wind_gen = df['Gen_Wind_Total'].sum()
total_curtail = df['Curtail_Total'].sum()

# 计算风光在总弃电中的比例
ratio_pv = np.divide(total_pv_gen, total_gen, out=np.zeros_like(total_pv_gen), where=total_gen > 0)
ratio_wind = np.divide(total_wind_gen, total_gen, out=np.zeros_like(total_wind_gen), where=total_gen > 0)

# 实际使用量 = 总发电量 - 被弃掉的量
total_pv_used = total_pv_gen - (total_curtail * ratio_pv)
total_wind_used = total_wind_gen - (total_curtail * ratio_wind)

# (4) 汇总计算联合园区的最终指标
total_load_kwh = df['Load_Total'].sum()
total_grid_buy_kwh = df['Grid_Buy_Total'].sum()
total_curtail_kwh = df['Curtail_Total'].sum()

# (5) 计算总成本
cost_grid = total_grid_buy_kwh * PRICE_GRID
cost_pv = total_pv_used * PRICE_PV
cost_wind = total_wind_used * PRICE_Wind
total_cost = cost_grid + cost_pv + cost_wind

# (6) 计算单位平均成本
avg_cost = total_cost / total_load_kwh


# --- 5. 打印结果 ---
if __name__ == "__main__":
    print("--- 问题2(1) 联合园区 (无储能) 经济性分析 ---")
    print(f"联合总购电量:       {total_grid_buy_kwh:,.2f} kWh")
    print(f"联合总弃风弃光电量: {total_curtail_kwh:,.2f} kWh")
    print(f"联合总供电成本:     {total_cost:,.2f} 元")
    print(f"联合单位平均成本:   {avg_cost:,.4f} 元/kWh")