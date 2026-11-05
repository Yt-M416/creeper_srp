# 向西极端时差结果的综合判断

## 一、结论

本轮探索将West 10–11 h的正向估计归类为：**赛事届次与国家构成共同影响下的不稳定观察性关联**。

这一判断并不表示某个国家或某一届赛事单独“造成”了结果。更准确的解释是：West极端组由不同国家在少数特定举办地形成，国家与赛事届次彼此交织；当模型同时控制两层构成后，原有正向估计明显衰减，且不再支持稳定的组间差异。现有结果不支持“向西旅行提高举重表现”的生理性解释。

## 二、证据链

### 1. ±12 h不是主要原因

完整样本中，West 10–11 h相对Minimal组的估计为β=5.165，95% CI [1.471, 8.860]，BH-FDR P=0.0307。排除精确±12 h后，β=5.593，95% CI [1.478, 9.709]，BH-FDR P=0.0386。方向和FDR结论均未改变，因此按预设规则，后续解释不再讨论边界问题。

### 2. West组在赛事届次层面较集中，但在国家层面并非最集中

- West 10–11 h的前三个国家为Thailand、Kazakhstan和USA，合计占58.1%，有效国家来源数为7.25；
- East 8–9 h前三个国家合计占77.5%，有效国家来源数为4.58；
- 因此，国家集中并不是West组独有的异常结构。

赛事届次层面的差异更明显：

- West 10–11 h前三届赛事合计占58.1%，有效赛事来源数为6.18；
- East 8–9 h前三届合计占42.6%，有效赛事来源数为9.91；
- East 10–11 h前三届合计占48.9%，有效赛事来源数为8.29。

West极端组因此比两个向东对照组更集中在少数赛事届次。

### 3. 2015 Houston是最强的正向驱动候选，但不是唯一来源

删除2015 Houston后，West估计由5.165降至2.361，Δβ=−2.804（−54.3%），BH-FDR P由0.0307变为0.6124。该届West组共23条记录，且只来自Thailand（13条）和Kazakhstan（10条）。

其他赛事的作用并不一致。删除2006 Santo Domingo、2009 Goyang、2007 Chiang Mai或2019 Pattaya后，West β反而升至6.385–7.054。这说明总体结果不是由所有赛事以同一方向推动，而是不同届次在增强或压低正向估计。

逐国家删除没有造成West系数变号。删除Kazakhstan、China、Iran、Thailand或Mexico后，β仍为正，但BH-FDR结论改变；删除Russia后β反而增加22.8%。国家影响主要表现为估计幅度与精度变化，而不是某一个国家完全决定效应方向。

### 4. 国家和赛事同时调整后，West估计明显衰减

| 模型 | West β | 95% CI | BH-FDR P | 相对基线变化 |
|---|---:|---:|---:|---:|
| 原主模型 | 5.165 | [1.471, 8.860] | 0.0307 | 0.0% |
| ＋国家固定效应 | 4.255 | [0.594, 7.916] | 0.1136 | −17.6% |
| ＋赛事届次固定效应 | 4.738 | [0.362, 9.114] | 0.1692 | −8.3% |
| ＋国家和赛事届次固定效应 | 2.199 | [−2.151, 6.549] | 0.5362 | −57.4% |

四个模型的设计矩阵均满秩且完成收敛。国家或赛事单独调整只造成中等程度衰减；二者同时调整后，West估计减少超过一半。这与“国家和赛事共同构成影响结果”的判断一致。

### 5. East 8–9 h的负向结果对单一来源删除更稳定

East 8–9 h的基线估计为β=−5.584，95% CI [−8.396, −2.773]，BH-FDR P=0.0010。在13次逐国家删除和22次逐赛事删除中，该估计始终保持负向且BH-FDR显著。国家调整后β=−3.990、BH-FDR P=0.0498；赛事届次调整后β=−7.124、BH-FDR P=0.0015。国家与赛事同时调整后方向仍为负（β=−3.923），但BH-FDR P升至0.1013。

因此，向东负向结果也会受充分构成调整影响，但它对任何单一国家或赛事的删除明显比West正向结果稳定。

## 三、历史背景的证据边界

IWF将2015 Houston列为Rio 2016资格赛。该届West极端组在本研究样本中只包含Thailand和Kazakhstan，而IWF公布的女子团体排名中两国分别列第6和第5。这个背景支持“强队构成与资格赛情境可能影响West估计”的解释。

2015 Houston还报告了大量反兴奋剂不利分析结果，其中涉及Kazakhstan运动员。但当前数据没有逐条连接最终裁决，也没有确认数据库是否按最终判罚更新。因此，反兴奋剂背景只能保留在内部核验记录中，不作为论文主解释，除非另行完成逐记录核对。

## 四、可用于Results的英文表述

> In post hoc explanatory analyses, the West 10–11 h category was more concentrated across championship editions than either eastward comparison category. Its three largest championship editions accounted for 58.1% of observations, compared with 42.6% for East 8–9 h and 48.9% for East 10–11 h. The westward estimate was sensitive to event composition. Excluding the 2015 World Championships in Houston reduced the coefficient from 5.17 to 2.36 Sinclair points, whereas no leave-one-nation-out or leave-one-championship-out analysis reversed its direction. Adding nation and championship-edition fixed effects together attenuated the estimate to 2.20 points (95% CI, −2.15 to 6.55; BH-FDR-adjusted P=0.536), a 57.4% reduction from the primary model.

如果正文篇幅有限，可压缩为：

> The positive estimate for West 10–11 h was sensitive to sample composition. It fell by 54.3% after exclusion of the 2015 Houston championships and by 57.4% after simultaneous adjustment for nation and championship edition (β=2.20; 95% CI, −2.15 to 6.55; BH-FDR-adjusted P=0.536).

## 五、可用于Discussion的英文表述

> The higher Sinclair Total observed after westward travel across 10–11 h should not be interpreted as a performance benefit of westward travel. Exploratory analyses indicated that this estimate partly reflected the combination of national-team and championship-edition composition. The largest attenuation followed exclusion of the 2015 World Championships in Houston, for which the West 10–11 h subgroup comprised only athletes from Thailand and Kazakhstan. The estimate also decreased by 57.4% when nation and championship edition were included together, and its confidence interval crossed zero. These results provide a data-based explanation for the unexpected direction of the association: the positive estimate was sensitive to who competed and where the championships were held. They do not identify a physiological mechanism.

若希望保留发现性研究的定位，可接：

> The finding is therefore better treated as a hypothesis-generating association than as evidence of improved adaptation to extreme westward travel. Sleep, circadian phase, acclimatization, departure and arrival timing, and training load were not measured, so the remaining association cannot be assigned to a specific biological or behavioral pathway.

## 六、可用于Limitations的英文表述

> Nation, host location, championship edition, and directional time-zone displacement were structurally related. Although post hoc composition analyses identified influential national and event clusters, they could not fully separate contextual effects from the travel exposure itself. The positive West 10–11 h estimate was substantially attenuated after simultaneous adjustment for nation and championship edition, which limits a physiological interpretation of this association. We also lacked sleep, circadian-phase, acclimatization, and detailed itinerary data needed to test a mechanism.

## 七、写作要求

- 选用IJSPP优先的sport-performance语体：直接报告估计值、置信区间和FDR结果；
- 使用“association”“estimate”“was sensitive to composition”，不用“benefit”“caused”或“improved performance”；
- 不把FDR不显著写成“没有效应”，应写成估计衰减且不再提供稳定差异证据；
- 不把2015 Houston的反兴奋剂背景写成已验证机制；
- 不把±12 h写入后续主解释，因为预设敏感性检验没有改变方向和FDR结论；
- 上述数字、模型含义和方向均来自本轮输出，未根据叙事需要改动。
