import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt  # 1. 导入图表库

# --- (新增) 设置 Matplotlib 中文字体 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置失败 (可能未安装 'SimHei')，图表标签可能显示异常: {e}")
# --- 结束新增 ---

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
df['Gen_PV_A'] = df['PV_A_pu'] * CAP_PV_A
df['Gen_Wind_B'] = df['Wind_B_pu'] * CAP_Wind_B
df['Gen_PV_C'] = df['PV_C_pu'] * CAP_PV_C
df['Gen_Wind_C'] = df['Wind_C_pu'] * CAP_Wind_C
df['Load_Total'] = df['Load_A'] + df['Load_B'] + df['Load_C']
df['Gen_PV_Total'] = df['Gen_PV_A'] + df['Gen_PV_C']
df['Gen_Wind_Total'] = df['Gen_Wind_B'] + df['Gen_Wind_C']
df['Gen_Total'] = df['Gen_PV_Total'] + df['Gen_Wind_Total']

# --- 4. 联合园区 “无储能” 经济性分析 ---
df['Net_Load_Total'] = df['Load_Total'] - df['Gen_Total']
df['Grid_Buy_Total'] = np.where(df['Net_Load_Total'] > 0, df['Net_Load_Total'], 0)
df['Curtail_Total'] = np.where(df['Net_Load_Total'] <= 0, -df['Net_Load_Total'], 0)
total_gen = df['Gen_Total'].sum()
total_pv_gen = df['Gen_PV_Total'].sum()
total_wind_gen = df['Gen_Wind_Total'].sum()
total_curtail = df['Curtail_Total'].sum()
ratio_pv = np.divide(total_pv_gen, total_gen, out=np.zeros_like(total_pv_gen), where=total_gen > 0)
ratio_wind = np.divide(total_wind_gen, total_gen, out=np.zeros_like(total_wind_gen), where=total_gen > 0)
total_pv_used = total_pv_gen - (total_curtail * ratio_pv)
total_wind_used = total_wind_gen - (total_curtail * ratio_wind)
total_load_kwh = df['Load_Total'].sum()
total_grid_buy_kwh = df['Grid_Buy_Total'].sum()
total_curtail_kwh = df['Curtail_Total'].sum()
cost_grid = total_grid_buy_kwh * PRICE_GRID
cost_pv = total_pv_used * PRICE_PV
cost_wind = total_wind_used * PRICE_Wind
total_cost = cost_grid + cost_pv + cost_wind
avg_cost = total_cost / total_load_kwh


# --- 5. (!! 新增模块 !!) 可视化函数 ---
def plot_joint_operation_timeseries(df_plot):
    """
    生成并保存 联合园区24小时功率平衡图
    """
    try:
        fig, ax = plt.subplots(figsize=(15, 7), layout='constrained')

        x = np.arange(24)  # 0到23小时

        # 1. 绘制 负荷 和 发电 曲线
        ax.plot(x, df_plot['Load_Total'], label='联合总负荷', color='black', linewidth=2, marker='o', markersize=4)
        ax.plot(x, df_plot['Gen_Total'], label='联合总发电', color='green', linewidth=2, linestyle='--', marker='^',
                markersize=4)

        # 2. 绘制 购电 和 弃电 区域
        # 购电 (净负荷 > 0)
        ax.fill_between(x, df_plot['Grid_Buy_Total'], color='red', alpha=0.3, label='从主网购电 (kW)')
        # 弃电 (净负荷 < 0)
        ax.fill_between(x, df_plot['Curtail_Total'], color='gray', alpha=0.3, label='弃风弃光 (kW)')

        ax.set_ylabel('功率 (kW)')
        ax.set_xlabel('时间 (小时)')
        ax.set_title('问题2(1): 联合园区24小时功率平衡图 (无储能)')
        ax.set_xticks(x)  # 显示所有小时
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')

        plt.savefig("plot_Q2_1_timeseries.png")
        print("\n--- 可视化图表已生成 ---")
        print("已保存 'plot_Q2_1_timeseries.png'")

    except Exception as e:
        print(f"\n--- 可视化图表生成失败 ---")
        print(f"错误: {e}")
        print("请确保已安装 matplotlib (pip install matplotlib) 并且中文字体 'SimHei' 已安装。")


# --- (新增模块结束) ---


# --- 6. 打印结果 (原第5步) ---
if __name__ == "__main__":
    print("--- 问题2(1) 联合园区 (无储能) 经济性分析 ---")
    print(f"联合总购电量:       {total_grid_buy_kwh:,.2f} kWh")
    print(f"联合总弃风弃光电量: {total_curtail_kwh:,.2f} kWh")
    print(f"联合总供电成本:     {total_cost:,.2f} 元")
    print(f"联合单位平均成本:   {avg_cost:,.4f} 元/kWh")

    # --- (新增) 调用绘图函数 ---
    plot_joint_operation_timeseries(df)