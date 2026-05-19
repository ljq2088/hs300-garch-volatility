"""
04_garch_models.py
GARCH模型估计：主模型 GARCH(1,1)-normal
+ 稳健性: Student-t GARCH, GJR-GARCH, AR(1)-GARCH。
输出参数表包含标准误、t值、p值。
保存标准化残差供诊断使用。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model
from tabulate import tabulate

PROJECT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
TABLES = PROJECT / "tables"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_returns():
    df = pd.read_csv(DATA_PROCESSED / "hs300_returns.csv", parse_dates=["date"])
    return df


def half_life(alpha_beta):
    if alpha_beta >= 1.0:
        return np.inf
    return np.log(0.5) / np.log(alpha_beta)


def _extract_fit_info(fit, model_name, model_type="garch"):
    """从 arch fit 对象提取公共信息。

    model_type: "garch", "gjr", "egarch"
    """
    p = fit.params
    se = fit.std_err
    tv = fit.tvalues
    pv = fit.pvalues

    omega = p["omega"]
    alpha = p["alpha[1]"]
    beta = p["beta[1]"]
    mu = p.get("mu", np.nan)

    is_asymmetric = model_type in ("gjr", "egarch")
    gamma = p["gamma[1]"] if is_asymmetric else None

    if model_type == "egarch":
        # EGARCH in log-variance space: persistence proxy = beta
        pers = beta
    elif model_type == "gjr":
        pers = alpha + beta + gamma / 2.0
    else:
        pers = alpha + beta

    hl = half_life(pers)

    def _pv(key):
        return pv.get(key, np.nan)

    return {
        "model": model_name,
        "model_type": model_type,
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "alpha_beta": pers,
        "half_life": hl,
        "loglik": fit.loglikelihood,
        "aic": fit.aic,
        "bic": fit.bic,
        "fit": fit,
        "cond_vol": fit.conditional_volatility,
        "std_resid": fit.std_resid,
        "params": p,
        "std_err": se,
        "tvalues": tv,
        "pvalues": pv,
        "p_mu": _pv("mu"),
        "p_omega": _pv("omega"),
        "p_alpha": _pv("alpha[1]"),
        "p_beta": _pv("beta[1]"),
        "p_gamma": _pv("gamma[1]") if is_asymmetric else None,
    }


def fit_garch_normal(ret_vals):
    """主模型: 常数均值 + GARCH(1,1) + 正态创新"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off")
    return _extract_fit_info(fit, "GARCH(1,1)-Normal", model_type="garch")


def fit_garch_t(ret_vals):
    """稳健性模型1: Student-t GARCH(1,1)"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="t", rescale=False)
    fit = model.fit(disp="off")
    r = _extract_fit_info(fit, "GARCH(1,1)-Student-t", model_type="garch")
    r["nu"] = fit.params.get("nu", np.nan)
    return r


def fit_gjr_garch(ret_vals):
    """稳健性模型2: GJR-GARCH(1,1)"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off")
    return _extract_fit_info(fit, "GJR-GARCH(1,1)-Normal", model_type="gjr")


def fit_ar1_garch(ret_vals):
    """稳健性模型3: AR(1) 均值 + GARCH(1,1) + 正态创新"""
    model = arch_model(ret_vals, mean="AR", lags=1, vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off")
    r = _extract_fit_info(fit, "AR(1)-GARCH(1,1)-Normal", model_type="garch")
    r["ar1"] = fit.params.get("y[1]", np.nan)
    return r


def fit_gjr_garch_t(ret_vals):
    """进一步稳健性: GJR-GARCH(1,1) + Student-t 创新"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                       dist="t", rescale=False)
    fit = model.fit(disp="off")
    r = _extract_fit_info(fit, "GJR-GARCH(1,1)-Student-t", model_type="gjr")
    r["nu"] = fit.params.get("nu", np.nan)
    return r


def fit_egarch_normal(ret_vals):
    """进一步稳健性: EGARCH(1,1) + 正态创新"""
    model = arch_model(ret_vals, mean="Constant", vol="EGARCH", p=1, o=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off")
    return _extract_fit_info(fit, "EGARCH(1,1)-Normal", model_type="egarch")


def fit_egarch_t(ret_vals):
    """进一步稳健性: EGARCH(1,1) + Student-t 创新"""
    model = arch_model(ret_vals, mean="Constant", vol="EGARCH", p=1, o=1, q=1,
                       dist="t", rescale=False)
    fit = model.fit(disp="off")
    r = _extract_fit_info(fit, "EGARCH(1,1)-Student-t", model_type="egarch")
    r["nu"] = fit.params.get("nu", np.nan)
    return r


def save_main_params(result, dates):
    """保存主模型参数表（含标准误、t值、p值）。"""
    r = result
    p = r["params"]
    se = r["std_err"]
    tv = r["tvalues"]
    pv = r["pvalues"]

    def _val(key):
        return p.get(key, np.nan)

    def _se(key):
        return se.get(key, np.nan)

    def _t(key):
        return tv.get(key, np.nan)

    def _p(key):
        return pv.get(key, np.nan)

    rows = [
        {"parameter": "mu",    "estimate": _val("mu"),       "std_error": _se("mu"),
         "t_stat": _t("mu"),       "p_value": _p("mu"),      "description": "Mean equation constant"},
        {"parameter": "omega", "estimate": _val("omega"),    "std_error": _se("omega"),
         "t_stat": _t("omega"),    "p_value": _p("omega"),   "description": "Baseline variance"},
        {"parameter": "alpha", "estimate": _val("alpha[1]"), "std_error": _se("alpha[1]"),
         "t_stat": _t("alpha[1]"), "p_value": _p("alpha[1]"),"description": "News impact (ARCH effect)"},
        {"parameter": "beta",  "estimate": _val("beta[1]"),  "std_error": _se("beta[1]"),
         "t_stat": _t("beta[1]"),  "p_value": _p("beta[1]"), "description": "Volatility persistence (GARCH effect)"},
        # Summary rows (no SE/t/p)
        {"parameter": "alpha+beta", "estimate": r["alpha_beta"], "std_error": "",
         "t_stat": "", "p_value": "", "description": "Volatility persistence index"},
        {"parameter": "half-life (days)", "estimate": r["half_life"], "std_error": "",
         "t_stat": "", "p_value": "", "description": "Shock half-life"},
        {"parameter": "Log-likelihood", "estimate": r["loglik"], "std_error": "",
         "t_stat": "", "p_value": "", "description": "Maximized log-likelihood"},
        {"parameter": "AIC", "estimate": r["aic"], "std_error": "",
         "t_stat": "", "p_value": "", "description": "Akaike information criterion"},
        {"parameter": "BIC", "estimate": r["bic"], "std_error": "",
         "t_stat": "", "p_value": "", "description": "Bayesian information criterion"},
    ]

    tbl = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TABLES / "table3_garch_main.csv", index=False)
    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".6f")
    (TABLES / "table3_garch_main.md").write_text(
        "Table 3: GARCH(1,1)-Normal parameter estimates with standard errors\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table3_garch_main saved.")

    # 保存条件波动率
    cv = r["cond_vol"]
    if hasattr(cv, "values"):
        cv = cv.values
    cond_vol = pd.DataFrame({"date": dates, "conditional_volatility": cv})
    cond_vol.to_csv(DATA_PROCESSED / "hs300_garch_conditional_volatility.csv", index=False)
    print("[INFO] conditional volatility saved.")

    # 保存标准化残差 — 来自 fit.std_resid，不手动计算
    sr = r["std_resid"]
    if hasattr(sr, "values"):
        sr = sr.values
    std_resid_df = pd.DataFrame({"date": dates, "std_resid": sr})
    std_resid_df.to_csv(DATA_PROCESSED / "hs300_garch_standardized_residuals.csv", index=False)
    print("[INFO] standardized residuals saved.")


def _gamma_significance_text(gamma, p_gamma):
    """根据 gamma 值和 p 值生成严谨的解释文本。
    对于 GJR-GARCH: gamma > 0 表示坏消息放大波动。
    对于 EGARCH: gamma < 0 表示杠杆效应（坏消息放大波动）。
    """
    if gamma is None or np.isnan(p_gamma):
        return "N/A"
    has_leverage = gamma != 0  # non-zero asymmetry in the expected direction
    if has_leverage and p_gamma < 0.05:
        direction = "positive" if gamma > 0 else "negative"
        return f"significant ({direction} asymmetry, p<0.05)"
    if has_leverage and p_gamma < 0.10:
        direction = "positive" if gamma > 0 else "negative"
        return f"weak/marginal ({direction} asymmetry, p<0.10)"
    if p_gamma >= 0.10:
        direction = "positive" if gamma > 0 else "negative"
        return f"{direction} but not statistically significant (p={p_gamma:.4f})"
    return "N/A"


def save_comparison_table(results):
    """保存模型比较表，含显著性判断。"""
    rows = []
    for r in results:
        gamma = r.get("gamma")
        p_gamma = r.get("p_gamma")
        notes = ""
        if r["model"].startswith("GJR"):
            gtext = _gamma_significance_text(gamma, p_gamma)
            notes = f"gamma={gamma:.6f}, p_gamma={p_gamma:.4f}: {gtext}"
        if "Student-t" in r["model"]:
            nu = r.get("nu", np.nan)
            notes = f"nu={nu:.4f} (heavy-tail dof)"
        if r["model"].startswith("AR(1)"):
            ar1 = r.get("ar1", np.nan)
            notes = f"AR(1) coeff={ar1:.6f}"

        rows.append({
            "model": r["model"],
            "distribution": "Normal" if "Normal" in r["model"] else "Student-t",
            "loglikelihood": r["loglik"],
            "AIC": r["aic"],
            "BIC": r["bic"],
            "omega": r["omega"],
            "alpha": r["alpha"],
            "gamma": gamma if gamma is not None else "",
            "gamma_pvalue": f"{p_gamma:.4f}" if p_gamma is not None and not np.isnan(p_gamma) else "",
            "beta": r["beta"],
            "alpha_beta_persistence": r["alpha_beta"],
            "half_life_days": f"{r['half_life']:.1f}" if r["half_life"] != np.inf else "inf",
            "notes": notes,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table4_model_comparison.csv", index=False)
    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".6f")
    (TABLES / "table4_model_comparison.md").write_text(
        "Table 4: Model comparison — GARCH(1,1) Normal, Student-t, GJR-GARCH, AR(1)-GARCH\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table4_model_comparison saved.")


def save_extended_asymmetric_table(results):
    """保存进一步稳健性非对称模型比较表 (Table 6)。"""
    rows = []
    for r in results:
        gamma = r.get("gamma")
        p_gamma = r.get("p_gamma")
        nu = r.get("nu", None)
        model_type = r.get("model_type", "garch")

        gtext = _gamma_significance_text(gamma, p_gamma)

        notes_parts = []
        if model_type == "egarch":
            notes_parts.append("EGARCH persistence proxy = |beta| (log-variance space)")
        if nu is not None and not np.isnan(nu):
            notes_parts.append(f"nu={nu:.4f}")
        notes_parts.append(f"asymmetry: {gtext}")
        notes = "; ".join(notes_parts)

        rows.append({
            "model": r["model"],
            "distribution": "Normal" if "Normal" in r["model"] else "Student-t",
            "loglikelihood": r["loglik"],
            "AIC": r["aic"],
            "BIC": r["bic"],
            "omega": r["omega"],
            "alpha": r["alpha"],
            "beta": r["beta"],
            "gamma": gamma if gamma is not None else "",
            "gamma_pvalue": f"{p_gamma:.4f}" if p_gamma is not None and not np.isnan(p_gamma) else "",
            "nu": f"{nu:.4f}" if nu is not None and not np.isnan(nu) else "",
            "vol_persistence_proxy": r["alpha_beta"],
            "half_life_days": f"{r['half_life']:.1f}" if r["half_life"] != np.inf else "inf",
            "notes": notes,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table6_extended_asymmetric_models.csv", index=False)
    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".6f")
    (TABLES / "table6_extended_asymmetric_models.md").write_text(
        "Table 6: Extended asymmetric volatility models — GJR-GARCH & EGARCH with Normal/Student-t\n\n"
        + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table6_extended_asymmetric_models saved.")


def plot_conditional_volatility(dates, main_result):
    """条件波动率图"""
    cv = main_result["cond_vol"]
    if hasattr(cv, "values"):
        cv = cv.values
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, cv, linewidth=0.4, color="black")
    ax.set_title("HS300 Conditional Volatility — GARCH(1,1)-Normal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Conditional volatility")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_garch_conditional_volatility.png", dpi=150)
    plt.close(fig)
    print("[INFO] fig5 saved.")


def main():
    df = load_returns()
    ret_vals = df["ret"].values
    dates = df["date"].values

    print("[INFO] Fitting GARCH(1,1)-Normal ...")
    garch_n = fit_garch_normal(ret_vals)
    print("[INFO] Fitting GARCH(1,1)-Student-t ...")
    garch_t = fit_garch_t(ret_vals)
    print("[INFO] Fitting GJR-GARCH(1,1) ...")
    gjr = fit_gjr_garch(ret_vals)
    print("[INFO] Fitting AR(1)-GARCH(1,1) ...")
    ar1g = fit_ar1_garch(ret_vals)
    print("[INFO] Fitting GJR-GARCH(1,1)-Student-t ...")
    gjr_t = fit_gjr_garch_t(ret_vals)
    print("[INFO] Fitting EGARCH(1,1)-Normal ...")
    egarch_n = fit_egarch_normal(ret_vals)
    print("[INFO] Fitting EGARCH(1,1)-Student-t ...")
    egarch_t = fit_egarch_t(ret_vals)

    # 保存主模型参数（含标准化残差）
    save_main_params(garch_n, dates)

    # 模型比较
    save_comparison_table([garch_n, garch_t, gjr, ar1g])

    # 进一步稳健性：非对称模型比较 (Table 6)
    asymmetric_models = [gjr, gjr_t, egarch_n, egarch_t]
    save_extended_asymmetric_table(asymmetric_models)

    # 条件波动率图
    plot_conditional_volatility(dates, garch_n)

    # 输出关键解释
    print("\n--- GARCH(1,1)-Normal Key Results ---")
    print(f"  mu      = {garch_n['mu']:.4f}  (p={garch_n['p_mu']:.4f})")
    print(f"  omega   = {garch_n['omega']:.6f}  (p={garch_n['p_omega']:.4f})")
    print(f"  alpha   = {garch_n['alpha']:.4f}  (p={garch_n['p_alpha']:.4f})")
    print(f"  beta    = {garch_n['beta']:.4f}  (p={garch_n['p_beta']:.4f})")
    print(f"  alpha+beta = {garch_n['alpha_beta']:.4f}")
    print(f"  half-life  = {garch_n['half_life']:.1f} days"
          if garch_n["half_life"] != np.inf else "  half-life  = inf (persistence >= 1)")
    print(f"  AIC     = {garch_n['aic']:.2f}")
    print(f"  BIC     = {garch_n['bic']:.2f}")

    print("\n--- Student-t GARCH(1,1) ---")
    print(f"  nu      = {garch_t.get('nu', np.nan):.4f}  (degrees of freedom)")
    print(f"  AIC     = {garch_t['aic']:.2f}")

    print("\n--- GJR-GARCH(1,1) ---")
    gjr_gamma = gjr["gamma"]
    gjr_p = gjr["p_gamma"]
    print(f"  gamma   = {gjr_gamma:.4f}")
    print(f"  p_gamma = {gjr_p:.4f}")
    print(f"  interpretation: {_gamma_significance_text(gjr_gamma, gjr_p)}")

    print("\n--- AR(1)-GARCH(1,1) ---")
    print(f"  AR(1) coeff = {ar1g.get('ar1', np.nan):.4f}")
    print(f"  AIC         = {ar1g['aic']:.2f}")

    print("\n--- Further Robustness: Asymmetric Models ---")
    for r in asymmetric_models:
        gamma = r.get("gamma")
        p_gamma = r.get("p_gamma")
        nu = r.get("nu", None)
        print(f"\n  {r['model']}:")
        print(f"    AIC = {r['aic']:.2f}, BIC = {r['bic']:.2f}")
        if nu is not None and not np.isnan(nu):
            print(f"    nu = {nu:.4f}")
        print(f"    gamma = {gamma:.6f}, p_gamma = {p_gamma:.4f}" if gamma is not None else "    gamma = N/A")
        print(f"    interpretation: {_gamma_significance_text(gamma, p_gamma)}")
        print(f"    persistence proxy = {r['alpha_beta']:.4f}")


if __name__ == "__main__":
    main()
