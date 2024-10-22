import numpy as np
import pandas as pd
import chardet

csv_path = "D:\\Python\\Math_Code\\data_csv.csv"
csv = pd.read_csv(csv_path,encoding="GB2312")
csv_df = pd.DataFrame(data=csv, index=None, columns=None, dtype=None, copy=False)
csv_df_intersection=csv_df[csv_df["交叉口"] == '经中路-纬中路']
# csv_df_time = csv_df.groupby("时间").sum()
# print(csv_df_time)
csv_df_time41 =  csv_df[csv_df["时间"] < '2024-04-02T00:00:00.000']
csv_df_time41_rank = csv_df_time41.groupby("时间").sum()
print(csv_df_time41_rank)

