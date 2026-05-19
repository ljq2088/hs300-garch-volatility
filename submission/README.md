# 高级时间序列分析课程写作作业 — 提交材料

## 提交人

赖景祺

## 论文题目

中国股票市场波动聚集与风险预警：基于沪深300指数收益率的GARCH模型分析

## 提交文件清单

```
submission/
├── README.md                  # 本说明文件
├── assignment_final.pdf       # 论文 PDF
├── assignment_final.tex       # 论文 LaTeX 源文件（可编辑）
├── ai_usage_statement.md      # AI 使用声明
├── code/                      # 可复现代码
│   ├── 00_download_data.py    # 数据下载（AkShare → 东方财富）
│   ├── 01_prepare_returns.py  # 收益率计算
│   ├── 02_descriptive_analysis.py  # 描述性统计与图表
│   ├── 03_tests.py            # ADF / Ljung-Box / ARCH-LM 检验
│   ├── 04_garch_models.py     # GARCH(1,1) + 稳健性模型
│   ├── 05_diagnostics.py      # 标准化残差诊断
│   ├── 06_generate_summary_outputs.py  # 汇总表格与验证
│   ├── run_all.sh             # 一键运行脚本
│   └── requirements.txt       # Python 依赖
└── data/                      # 数据文件
    ├── raw/
    │   └── hs300_daily_raw.csv           # 原始下载数据
    └── processed/
        ├── hs300_daily_clean.csv         # 标准化清理数据
        └── hs300_returns.csv             # 日对数收益率（2002-01-07 至 2026-05-11，5902 观测）
```

运行时还会生成以下目录（无需手动创建）：
- `figures/` — 9 张 PNG 图表
- `tables/` — CSV 和 Markdown 格式的 6 组参数与诊断表格
- `data/processed/` 下的 `hs300_garch_conditional_volatility.csv` 和 `hs300_garch_standardized_residuals.csv`

## 复现说明

### 环境配置

```bash
conda create -n ts_garch python=3.11 -y
conda activate ts_garch
pip install -r code/requirements.txt
```

### 运行

```bash
bash code/run_all.sh
```

或逐步运行：

```bash
python code/00_download_data.py
python code/01_prepare_returns.py
python code/02_descriptive_analysis.py
python code/03_tests.py
python code/04_garch_models.py
python code/05_diagnostics.py
python code/06_generate_summary_outputs.py
```

生成的图表和表格保存在当前工作目录下的 `figures/` 和 `tables/` 目录。

### 论文 PDF 编译

```bash
xelatex assignment_final.tex
xelatex assignment_final.tex   # 第二次以获得正确的交叉引用
```

需要系统安装 SimSun 字体（Windows 自带）及 TeX Live 2026。

## 数据来源

沪深300指数日度行情数据，来自东方财富，通过 AkShare Python 包获取。若 `index_zh_a_hist` 接口失败，脚本自动回退至 `stock_zh_index_daily` 接口。

## AI 使用声明

见 `ai_usage_statement.md`。
