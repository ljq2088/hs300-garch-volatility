"""
04_garch_models.py
GARCH模型估计：主模型 GARCH(1,1)-normal + 稳健性 Student-t 和 GJR-GARCH。
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


def fit_garch_normal(ret_vals):
    """主模型: 常数均值 + GARCH(1,1) + 正态创新"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off")
    params = fit.params
    omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
    mu = params.get("mu", np.nan)
    pers = alpha + beta
    hl = half_life(pers)

    return {
        "model": "GARCH(1,1)-Normal",
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": None,
        "alpha_beta": pers,
        "half_life": hl,
        "loglik": fit.loglikelihood,
        "aic": fit.aic,
        "bic": fit.bic,
        "fit": fit,
        "cond_vol": fit.conditional_volatility,
        "std_resid": fit.std_resid,
    }


def fit_garch_t(ret_vals):
    """稳健性模型1: Student-t GARCH(1,1)"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="t", rescale=False)
    fit = model.fit(disp="off")
    params = fit.params
    omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
    mu = params.get("mu", np.nan)
    nu = params.get("nu", np.nan)
    pers = alpha + beta
    hl = half_life(pers)

    return {
        "model": "GARCH(1,1)-Student-t",
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": None,
        "alpha_beta": pers,
        "half_life": hl,
        "loglik": fit.loglikelihood,
        "aic": fit.aic,
        "bic": fit.bic,
        "fit": fit,
        "cond_vol": fit.conditional_volatility,
        "std_resid": fit.std_resid,
        "nu": nu,
    }


def fit_gjr_garch(ret_vals):
    """稳健性模型2: GJR-GARCH(1,1)"""
    model = arch_model(ret_vals, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                       dist="normal", rescale=False)
    fit = model.fit(disp="off")
    params = fit.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    gamma = params["gamma[1]"]
    beta = params["beta[1]"]
    mu = params.get("mu", np.nan)
    pers = alpha + beta + gamma / 2.0  # expected persistence under symmetry
    hl = half_life(pers) if pers < 1 else np.inf

    return {
        "model": "GJR-GARCH(1,1)-Normal",
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
    }


def save_main_params(result, dates):
    """保存主模型参数表"""
    r = result
    tbl = pd.DataFrame([
        {"parameter": "mu", "value": r["mu"], "description": "Mean equation constant"},
        {"parameter": "omega", "value": r["omega"], "description": "Baseline variance"},
        {"parameter": "alpha", "value": r["alpha"], "description": "News impact (ARCH effect)"},
        {"parameter": "beta", "value": r["beta"], "description": "Volatility persistence (GARCH effect)"},
        {"parameter": "alpha + beta", "value": r["alpha_beta"], "description": "Volatility persistence index"},
        {"parameter": "half-life (days)", "value": r["half_life"], "description": "Shock half-life"},
        {"parameter": "Log-likelihood", "value": r["loglik"], "description": "Maximized log-likelihood"},
        {"parameter": "AIC", "value": r["aic"], "description": "Akaike information criterion"},
        {"parameter": "BIC", "value": r["bic"], "description": "Bayesian information criterion"},
    ])

    TABLES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TABLES / "table3_garch_main.csv", index=False)
    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".6f")
    (TABLES / "table3_garch_main.md").write_text(
        "Table 3: GARCH(1,1) main model parameter estimates\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table3_garch_main saved.")

    # 保存条件波动率
    cv = r["cond_vol"]
    if hasattr(cv, "values"):
        cv = cv.values
    cond_vol = pd.DataFrame({
        "date": dates,
        "conditional_volatility": cv,
    })
    cond_vol.to_csv(DATA_PROCESSED / "hs300_garch_conditional_volatility.csv", index=False)
    print("[INFO] conditional volatility saved.")


def save_comparison_table(results):
    """保存模型比较表"""
    rows = []
    for r in results:
        notes = ""
        if r["gamma"] is not None:
            notes = f"gamma = {r['gamma']:.6f}"
            if r["gamma"] > 0:
                notes += " (bad-news leverage effect detected)"
            else:
                notes += " (asymmetry not significant or opposite sign)"
        if "Student-t" in r["model"]:
            nu = r.get("nu", np.nan)
            notes = f"nu = {nu:.4f} (heavy-tail dof)"
        rows.append({
            "model": r["model"],
            "distribution": "Normal" if "Normal" in r["model"] else "Student-t",
            "loglikelihood": r["loglik"],
            "AIC": r["aic"],
            "BIC": r["bic"],
            "omega": r["omega"],
            "alpha": r["alpha"],
            "gamma": r["gamma"] if r["gamma"] is not None else "",
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
        "Table 4: Model comparison — GARCH(1,1) Normal, Student-t, GJR-GARCH\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table4_model_comparison saved.")


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

    # 保存主模型参数
    save_main_params(garch_n, dates)

    # 模型比较
    save_comparison_table([garch_n, garch_t, gjr])

    # 条件波动率图
    plot_conditional_volatility(dates, garch_n)

    # 输出关键解释
    print("\n--- GARCH(1,1)-Normal Key Results ---")
    print(f"  mu      = {garch_n['mu']:.4f}")
    print(f"  omega   = {garch_n['omega']:.6f}")
    print(f"  alpha   = {garch_n['alpha']:.4f}")
    print(f"  beta    = {garch_n['beta']:.4f}")
    print(f"  alpha+beta = {garch_n['alpha_beta']:.4f}")
    print(f"  half-life  = {garch_n['half_life']:.1f} days"
          if garch_n["half_life"] != np.inf else "  half-life  = inf (persistence >= 1)")
    print(f"  AIC     = {garch_n['aic']:.2f}")
    print(f"  BIC     = {garch_n['bic']:.2f}")

    print("\n--- Student-t GARCH(1,1) ---")
    print(f"  nu      = {garch_t.get('nu', np.nan):.4f}  (degrees of freedom)")
    print(f"  AIC     = {garch_t['aic']:.2f}")

    print("\n--- GJR-GARCH(1,1) ---")
    print(f"  gamma   = {gjr['gamma']:.4f}")
    sign = "bad-news > good-news" if gjr["gamma"] > 0 else "opposite/negligible"
    print(f"  interpretation: {sign}")


if __name__ == "__main__":
    main()
