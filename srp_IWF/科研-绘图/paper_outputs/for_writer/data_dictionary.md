# 数据字典（核心列）

| 列名 | 说明 |
|------|------|
| `Athlete_ID` | 运动员唯一标识 |
| `Event` | 比赛名称（含 Men/Women 用于解析性别） |
| `Total` | 总成绩 (kg) |
| `Body_Weight` | 体重 (kg) |
| `Athlete_UTC_offset` / `Venue_UTC_offset` | 运动员居住地 / 赛场 UTC 偏移 (h) |
| `JetLag_signed` | 圆形 24h 最短时差，东向为正 (h) |
| `JetLagGroup_dir` | 方向性 11 档分组；参照组 `Minimal(0-1h)` |
| `Sinclair_Total` | IWF Sinclair 标准化总分（跨级别可比） |
| `TravelDuration` | 估计旅行时长 (h) |
| `Reference_Date` | 比赛参考日期 |
| `Born` | 出生日期（原始字符串，解析用于年龄） |

详细预处理见仓库内 `LMM_jetlag.py`、`heterogeneity_analysis.py` 注释。
