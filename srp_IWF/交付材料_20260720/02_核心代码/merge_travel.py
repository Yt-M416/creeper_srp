"""
merge_travel.py
将运动员成绩数据与旅行时长数据合并，输出 athlete_top13_travel.csv。

用法：
    python merge_travel.py

输出：
    athlete_top13_travel.csv  （含 TravelDuration 列，缺失值留 NaN）
"""

import os
import pandas as pd

PERF_CSV   = r"D:\Code\Python\Creeper\srp_IWF\iwf_data_top13_athlete2plus_circular.csv"
TRAVEL_CSV = r"D:\Code\Python\Creeper\srp_IWF\travel_time_merged.csv"
OUTPUT_CSV = r"D:\Code\Python\Creeper\srp_IWF\athlete_top13_travel.csv"


def normalize_city(s: str) -> str:
    """统一城市名格式：去除逗号周围空格，转小写再还原，避免大小写差异。
    'Santo Domingo, DOM' → 'santo domingo,dom'
    'PATTAYA, THA'      → 'pattaya,tha'
    """
    if not isinstance(s, str):
        return s
    parts = [p.strip() for p in s.split(",")]
    return ",".join(parts).lower()


def main() -> None:
    # ---- 读取 ----
    df_perf = pd.read_csv(PERF_CSV, encoding="utf-8-sig")
    df_perf.columns = df_perf.columns.str.strip()

    df_travel = pd.read_csv(TRAVEL_CSV, encoding="utf-8-sig")
    df_travel.columns = df_travel.columns.str.strip()
    df_travel = df_travel.dropna(subset=["Departure_Nation"])

    print(f"Performance rows : {len(df_perf)}")
    print(f"Travel rows      : {len(df_travel)}")

    # ---- 规范化城市名（两侧都做，消除空格/大小写差异）----
    # 关键修正：主数据 'Santo Domingo, DOM' vs travel 'Santo Domingo,DOM'
    df_perf  ["City_key"] = df_perf  ["City"].apply(normalize_city)
    df_travel["City_key"] = df_travel["Arrival_City"].apply(normalize_city)

    # ---- 规范化 Nation（travel 数据已是 3 字母代码，主数据同）----
    df_travel = df_travel.rename(columns={
        "Departure_Nation":  "Nation",
        "Duration_Hours(h)": "TravelDuration",
        "Year":              "Event_Year",       # 统一到主数据的年份列名
    })

    # ---- 检查 travel 数据内是否有 (Nation, City_key, Event_Year) 重复 ----
    dup_mask = df_travel.duplicated(subset=["Nation", "City_key", "Event_Year"])
    if dup_mask.any():
        print(f"[WARN] travel 数据内 {dup_mask.sum()} 行有重复键，取均值合并")
        df_travel = (
            df_travel
            .groupby(["Nation", "City_key", "Event_Year"], as_index=False)
            ["TravelDuration"].mean()
        )

    # ---- Merge（left join，只取需要的列）----
    n_before = len(df_perf)
    df = pd.merge(
        df_perf,
        df_travel[["Nation", "City_key", "Event_Year", "TravelDuration"]],
        on=["Nation", "City_key", "Event_Year"],
        how="left",
    )
    n_after = len(df)

    if n_after != n_before:
        raise RuntimeError(
            f"Merge 产生重复行（{n_before} → {n_after}），请检查键唯一性！"
        )

    df = df.drop(columns=["City_key"])

    # ---- 缺失报告 ----
    n_miss  = df["TravelDuration"].isna().sum()
    n_total = len(df)
    print(f"\nMerge 完成：{n_total} 行，TravelDuration 缺失 {n_miss} ({n_miss/n_total*100:.1f}%)")

    if n_miss > 0:
        miss_detail = (
            df[df["TravelDuration"].isna()]
            .groupby(["Nation", "Event_Year"])
            .size()
            .reset_index(name="n_missing")
            .sort_values("n_missing", ascending=False)
        )
        print("\n缺失明细（Nation × Year）：")
        print(miss_detail.to_string(index=False))

        # 给出 travel 数据里哪些 (City_key, Year) 组合可能是原因
        perf_keys  = set(zip(df_perf["City"].apply(normalize_city), df_perf["Event_Year"]))
        trav_keys  = set(zip(df_travel["City_key"],                  df_travel["Event_Year"]))
        unmatched  = perf_keys - trav_keys
        if unmatched:
            print("\n主数据中未被 travel 数据覆盖的 (City, Year)：")
            for c, y in sorted(unmatched):
                print(f"  {c!r}  {y}")

    # ---- 保存 ----
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
