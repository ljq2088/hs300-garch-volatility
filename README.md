# 中国股票市场波动聚集与风险预警：基于沪深300指数收益率的GARCH模型分析

高级时间序列分析课程写作作业项目。

## 项目主题

使用GARCH族模型分析沪深300指数日收益率的波动聚集现象，研究中国股票市场的时变风险特征。**GARCH模型用于刻画条件波动率，不预测指数涨跌方向。**

## 数据来源

沪深300指数日度行情数据，通过AkShare从东方财富获取。样本区间2002-01-07至2026-05-11，共5902个交易日。

## 环境与安装

```bash
# 创建conda环境
conda create -n ts_garch python=3.11 -y
conda activate ts_garch
pip install -r requirements.txt
```

## 运行方式

### 一键运行

```bash
conda activate ts_garch
bash run_all.sh
```

### 逐步运行

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
    ├── hs300_garch_conditional_volatility.csv # GARCH条件波动率
    └── hs300_garch_standardized_residuals.csv # 标准化残差 (fit.std_resid)

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
├── table3_garch_main.csv/.md                  # GARCH(1,1)主模型参数（含SE/t/p）
├── table4_model_comparison.csv/.md            # 模型比较
├── table5_residual_diagnostics.csv/.md        # 残差诊断
└── table6_extended_asymmetric_models.csv/.md  # 扩展非对称模型（GJR/EGARCH × Normal/Student-t）

paper/
├── assignment_draft.md                        # 论文初稿
├── assignment_final.pdf                       # 论文PDF
├── ai_usage_statement.md                      # AI使用声明
└── beamer_presentation.tex                    # Beamer汇报PPT源文件
```

## 主要发现

| 维度 | 结果 |
|------|------|
| 样本 | 2002-01-07 ~ 2026-05-11，5902个交易日 |
| 波动聚集 | ARCH-LM lag=5: 统计量=529.63，p≈0 |
| GARCH(1,1) α | 0.0734 (p<0.001) — 新冲击对波动有显著正向影响 |
| GARCH(1,1) β | 0.9195 (p<0.001) — 条件方差惯性极强 |
| α+β | 0.9928 — 波动高度持久，半衰期≈96个交易日 |
| Student-t | ν=5.15，AIC=19901.63，远优于正态GARCH (AIC=20313.98) |
| GJR-GARCH-Normal | γ=0.0159，p=0.2172 — 为正但不显著，不能强断言存在杠杆效应 |
| GJR-GARCH-Student-t | γ=0.0217，p=0.0463 — **在厚尾分布下显著**，非对称效应获统计支持 |
| EGARCH-Student-t | γ=−0.0123，p=0.1007 — 接近10%边界但未达常用显著性水平，非对称证据较弱 |
| 残差诊断 | z_t² Ljung-Box和ARCH-LM均不显著，GARCH成功捕捉条件异方差 |

**注意**：GJR-GARCH在正态假设下非对称项不显著（p=0.217），但在Student-t设定下达到5%显著性（p=0.046）。EGARCH的非对称效应即使在Student-t下也仅接近10%边界（p=0.101），未达常用显著性水平。因此核心稳健结论是波动聚集、厚尾和高度持久性，非对称效应证据对模型形式有一定敏感性。

## 当前状态

代码、表格、图形、论文草稿和AI使用声明均已生成；最终提交前需人工检查PDF排版和图表编号。

## PDF 生成

`paper/assignment_draft.md` 是最终可编辑源文件，可通过以下方式转为PDF：
- [pandoc](https://pandoc.org)：`pandoc assignment_draft.md -o assignment_final.pdf --pdf-engine=xelatex -V CJKmainfont="SimSun"`
- [Typora](https://typora.io)：直接导出PDF
- VS Code + Markdown PDF 插件

## Beamer PPT 编译

`paper/beamer_presentation.tex` 为约20分钟的Beamer汇报PPT源文件。本地编译：

```bash
# 需要 xelatex (TeX Live)
xelatex beamer_presentation.tex
xelatex beamer_presentation.tex   # 第二次以获得正确的目录
```

若中文字体不同，修改 `.tex` 文件中 `\setCJKmainfont{SimSun}` 等为系统可用字体（如 Windows 用 `SimSun`/`SimHei`/`KaiTi`，macOS 用 `Songti SC`/`Heiti SC`，Linux 用 `Noto Serif CJK SC`）。

## 注意事项

1. 所有数值来自实际代码运行结果
2. 若AkShare接口1（`index_zh_a_hist`）失败，脚本自动回退到接口2（`stock_zh_index_daily`）
3. 两个接口都失败时会给出明确报错并提示手动提供CSV
4. 标准化残差来自arch包的`fit.std_resid`，不做手动`ret/sigma`计算
5. 参数表包含标准误、t值和p值，推断基于显著性而非系数符号
6. 图表中的中文字体可能需要根据系统自行调整
