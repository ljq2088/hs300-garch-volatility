# 中国股票市场波动聚集与风险预警：基于沪深300指数收益率的GARCH模型分析

高级时间序列分析课程写作作业项目。

## 项目主题

使用GARCH族模型分析沪深300指数日收益率的波动聚集现象，研究中国股票市场的时变风险特征。强调GARCH用于刻画条件波动率而非预测指数涨跌方向。

## 数据来源

沪深300指数日度行情数据，通过AkShare从东方财富获取，样本区间2002-01-07至2026-05-11，共5902个交易日。

## 环境与安装

```bash
# 创建conda环境
conda create -n ts_garch python=3.11 -y
conda activate ts_garch

# 安装依赖
pip install -r requirements.txt
```

## 运行流程

```bash
conda activate ts_garch
python src/00_download_data.py    # 数据下载与清洗
python src/01_prepare_returns.py  # 收益率计算
python src/02_descriptive_analysis.py  # 描述性统计与图表
python src/03_tests.py            # ADF, Ljung-Box, ARCH-LM
python src/04_garch_models.py     # GARCH(1,1) + 稳健性模型
python src/05_diagnostics.py      # 残差诊断
python src/06_generate_summary_outputs.py  # 汇总与验证
```

## 输出文件

```
data/
├── raw/hs300_daily_raw.csv                    # 原始下载数据
└── processed/
    ├── hs300_daily_clean.csv                  # 标准化清理数据
    ├── hs300_returns.csv                      # 日对数收益率
    └── hs300_garch_conditional_volatility.csv # GARCH条件波动率

figures/
├── fig1_hs300_close.png                       # 收盘价走势
├── fig2_hs300_returns.png                     # 收益率序列
├── fig3_hs300_squared_returns.png             # 平方收益率
├── fig4_return_histogram.png                  # 收益率直方图
├── fig5_garch_conditional_volatility.png      # 条件波动率
├── fig6_standardized_residuals.png            # 标准化残差
├── fig7_squared_standardized_residuals.png    # 平方标准化残差
├── fig8_acf_standardized_residuals.png        # 标准化残差ACF
└── fig9_acf_squared_standardized_residuals.png # 平方标准化残差ACF

tables/
├── table1_descriptive_stats.csv/.md           # 描述性统计
├── table2_tests.csv/.md                       # 时间序列检验
├── table3_garch_main.csv/.md                  # GARCH(1,1)主模型参数
├── table4_model_comparison.csv/.md            # 模型比较
└── table5_residual_diagnostics.csv/.md        # 残差诊断

paper/
└── assignment_draft.md                        # 论文初稿
```

## 主要发现

- 沪深300日收益率存在显著的波动聚集（ARCH-LM p≈0）
- GARCH(1,1) α+β=0.9928，波动持久性高，半衰期约96个交易日
- Student-t GARCH的AIC显著低于正态GARCH，确认厚尾特征
- GJR-GARCH γ=0.0159，非对称效应存在但幅度较小
- 标准化残差诊断确认GARCH成功捕捉条件异方差结构

## 注意事项

1. 所有数值来自实际代码运行结果，未手动编造
2. 若AkShare接口1（index_zh_a_hist）失败，脚本自动回退到接口2（stock_zh_index_daily）
3. 若两个接口都失败，脚本会给出明确报错并提示手动提供CSV
4. 论文中可以引用tables/中的实际数值
5. 图表中的中文字体可能需要根据系统自行调整
