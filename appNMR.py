import json, io, gc, numpy as np, pandas as pd, streamlit as st
import altair as alt

try:
    from numerapi.numerapi import NumerAPI
except Exception:  # fallback pour environnements où l'API est exposée différemment
    from numerapi import NumerAPI

st.set_page_config(page_title="Analyse Modèle Numerai", layout="wide")
st.title("Analyse Modèle Numerai")

# --- Patch de secours pour st.dataframe (pyarrow/numpy peut échouer à l'import) ---
try:
    _st_dataframe_orig = st.dataframe

    def _st_dataframe_safe(*args, **kwargs):
        try:
            return _st_dataframe_orig(*args, **kwargs)
        except Exception as e:
            st.warning(f"Affichage simplifié (pyarrow/numpy indisponible): {e}")
            try:
                obj = args[0] if args else None
                return st.write(obj)
            except Exception:
                return st.text("[Erreur d'affichage du DataFrame]")

    st.dataframe = _st_dataframe_safe
except Exception:
    pass

QUERY_V2 = """
query($modelId: String!) {
  v2RoundModelPerformances(modelId: $modelId) {
    atRisk
    churnThreshold
    corrMultiplier
    mmcMultiplier
    prevWeekChurnMax
    prevWeekTurnoverMax
    roundCloseStakingTime
    roundDataDatestamp
    roundNumber
    roundPayoutFactor
    roundResolveTime
    roundResolved
    roundScoreTime
    roundTarget
    tcMultiplier
    turnoverThreshold
    submissionScores {
      date
      day
      displayName
      payoutPending
      payoutSettled
      percentile
      resolveDate
      resolved
      value
    }
  }
}
"""


def _dt_naive(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    t = pd.to_datetime(x, errors="coerce")
    if pd.isna(t):
        return None
    try:
        if getattr(t, "tzinfo", None) is not None:
            return t.tz_localize(None)
    except Exception:
        pass
    return t


def _load_rounds_from_obj(raw):
    if (
        isinstance(raw, dict)
        and "data" in raw
        and "v2RoundModelPerformances" in raw["data"]
    ):
        return raw["data"]["v2RoundModelPerformances"]
    if isinstance(raw, list):
        return raw
    raise ValueError("JSON inattendu: structure non reconnue")


def build_long_df(rounds):
    rows = []
    for r in rounds:
        rd = (
            pd.to_datetime(str(r.get("roundDataDatestamp")))
            if r.get("roundDataDatestamp")
            else None
        )
        rst = _dt_naive(r.get("roundScoreTime")) if r.get("roundScoreTime") else None
        rrt = (
            _dt_naive(r.get("roundResolveTime")) if r.get("roundResolveTime") else None
        )
        payout = (
            float(r["roundPayoutFactor"])
            if r.get("roundPayoutFactor") is not None
            else None
        )
        at_risk = float(r["atRisk"]) if r.get("atRisk") is not None else None
        for s in r.get("submissionScores", []):
            rows.append(
                {
                    "roundNumber": r.get("roundNumber"),
                    "roundDate": rd,
                    "roundScoreTime": rst,
                    "roundResolveTime": rrt,
                    "roundPayoutFactor": payout,
                    "atRisk": at_risk,
                    "displayName": s.get("displayName"),
                    "value": s.get("value"),
                    "percentile": s.get("percentile"),
                    "day": s.get("day"),
                    "date": _dt_naive(s.get("date")) if s.get("date") else None,
                    "resolveDate": (
                        _dt_naive(s.get("resolveDate"))
                        if s.get("resolveDate")
                        else None
                    ),
                    "resolved": s.get("resolved"),
                }
            )
    df = pd.DataFrame(rows)
    for col in ["roundScoreTime", "roundResolveTime", "date", "resolveDate"]:
        if col in df.columns:
            try:
                tz = getattr(df[col].dt, "tz", None)
                if tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
            except Exception:
                pass
    # Downcast pour réduire la mémoire
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
    if "percentile" in df.columns:
        df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce").astype(
            "float32"
        )
    if "roundPayoutFactor" in df.columns:
        df["roundPayoutFactor"] = pd.to_numeric(
            df["roundPayoutFactor"], errors="coerce"
        ).astype("float32")
    if "atRisk" in df.columns:
        df["atRisk"] = pd.to_numeric(df["atRisk"], errors="coerce").astype("float32")
    if "roundNumber" in df.columns:
        df["roundNumber"] = pd.to_numeric(
            df["roundNumber"], errors="coerce", downcast="integer"
        )
    if "displayName" in df.columns:
        try:
            df["displayName"] = df["displayName"].astype("category")
        except Exception:
            pass
    if "day" in df.columns:
        try:
            df["day"] = df["day"].astype("category")
        except Exception:
            pass
    return df


def find_season_col(columns):
    for c in columns:
        if str(c).strip().lower().replace(" ", "_") == "season_score":
            return c
    return None


def pctile_stats(series: pd.Series):
    ssp = pd.to_numeric(series, errors="coerce")
    desc = ssp.describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    return desc.to_frame(name="value").reset_index().rename(columns={"index": "stat"})


def make_hist(series: pd.Series, bins: int = 20):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count", "bin_mid"])
    lo, hi = float(clean.min()), float(clean.max())
    if lo == hi:
        edges = np.array([lo, lo + 1e-9], dtype=float)
        hist = np.array([clean.size], dtype=int)
    else:
        edges = np.linspace(lo, hi, bins + 1)
        hist, _ = np.histogram(clean, bins=edges)
    hist_df = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}
    )
    hist_df["bin_mid"] = round((hist_df["bin_left"] + hist_df["bin_right"]) / 2, 5)
    return hist_df


def get_ssp_series(df: pd.DataFrame) -> pd.Series:
    if "season_score_payout" in df.columns:
        return pd.to_numeric(df["season_score_payout"], errors="coerce")
    return pd.Series(dtype=float)


def simulate_paths(
    clean_ssp: np.ndarray, L: int, n_paths: int, rng: np.random.Generator
):
    # Génère en float32 pour limiter la mémoire
    draws = rng.choice(
        np.asarray(clean_ssp, dtype=np.float32), size=(L, n_paths), replace=True
    )
    cumprod = np.vstack(
        [
            np.ones((1, n_paths), dtype=np.float32),
            np.cumprod(draws, axis=0).astype(np.float32),
        ]
    )
    steps = np.arange(L + 1, dtype=np.int32)
    cols = [f"path_{i+1}" for i in range(n_paths)]
    df = pd.DataFrame(cumprod, columns=cols)
    df.insert(0, "step", steps)
    for c in cols:
        df[c] = df[c].astype("float32")
    return df


def simulate_single_path(
    clean_ssp: np.ndarray, L: int, rng: np.random.Generator
) -> np.ndarray:
    """Simule une seule trajectoire cumulée (longueur L+1) avec point initial 1.0."""
    if L <= 0:
        return np.array([1.0], dtype=np.float32)
    draws = rng.choice(clean_ssp, size=L, replace=True)
    out = np.empty(L + 1, dtype=np.float32)
    out[0] = 1.0
    out[1:] = np.cumprod(draws).astype(np.float32)
    return out


def terminal_stats(
    clean_ssp: np.ndarray, horizons: list[int], n_sims: int, rng: np.random.Generator
):
    rows = []
    for h in horizons:
        if h <= 0:
            continue
        sim_draws = rng.choice(clean_ssp, size=(n_sims, h), replace=True)
        terminal = sim_draws.prod(axis=1)
        rows.append(
            {
                "horizon": h,
                "mean": float(np.mean(terminal)),
                "std": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
                "p1": float(np.percentile(terminal, 1)),
                "p5": float(np.percentile(terminal, 5)),
                "p25": float(np.percentile(terminal, 25)),
                "p50": float(np.percentile(terminal, 50)),
                "p75": float(np.percentile(terminal, 75)),
                "p95": float(np.percentile(terminal, 95)),
                "p99": float(np.percentile(terminal, 99)),
            }
        )
    return pd.DataFrame(rows)


def terminal_stats_stream(
    clean_ssp: np.ndarray,
    horizons: list[int],
    n_sims: int,
    rng: np.random.Generator,
    batch_size: int = 20000,
    sample_size: int = 50000,
):
    rows = []
    clean_ssp = np.asarray(clean_ssp, dtype=np.float32)
    for h in horizons:
        if h <= 0:
            continue
        # Welford pour mean/std
        n = 0
        mean = 0.0
        M2 = 0.0
        # Réservoir pour quantiles
        k = int(min(sample_size, n_sims))
        reservoir = np.empty(k, dtype=np.float32)
        filled = 0
        seen = 0
        remaining = n_sims
        while remaining > 0:
            b = int(min(batch_size, remaining))
            draws = rng.choice(clean_ssp, size=(b, h), replace=True)
            terminal = draws.prod(axis=1).astype(np.float32)
            # stats
            for x in terminal:
                seen += 1
                n += 1
                delta = float(x) - mean
                mean += delta / n
                M2 += delta * (float(x) - mean)
                if filled < k:
                    reservoir[filled] = x
                    filled += 1
                else:
                    j = rng.integers(0, seen)
                    if j < k:
                        reservoir[j] = x
            remaining -= b
        std = float(np.sqrt(M2 / (n - 1))) if n > 1 else 0.0
        # quantiles approximés depuis le réservoir
        q = (
            np.percentile(reservoir[:filled], [1, 5, 25, 50, 75, 95, 99])
            if filled > 0
            else [np.nan] * 7
        )
        rows.append(
            {
                "horizon": h,
                "mean": float(mean),
                "std": std,
                "p1": float(q[0]),
                "p5": float(q[1]),
                "p25": float(q[2]),
                "p50": float(q[3]),
                "p75": float(q[4]),
                "p95": float(q[5]),
                "p99": float(q[6]),
            }
        )
    return pd.DataFrame(rows)


def bisection_irr(cash_flows: np.ndarray, max_iter: int = 200, tol: float = 1e-7):
    def npv(r):
        return np.sum(cash_flows / (1 + r) ** np.arange(len(cash_flows)))

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid


def bisection_irr_stream(
    L: int,
    initial_balance: float,
    final_balance: float,
    contrib_amount: float,
    contrib_every: int,
    contrib_growth_rate: float,
    start_step: int | None,
    stop_step: int | None,
    max_iter: int = 200,
    tol: float = 1e-7,
):
    ss = start_step if start_step is not None else 1
    stp = stop_step if stop_step is not None else L

    def npv(r: float) -> float:
        # NPV = sum_{t=0..L} CF_t / (1+r)^t
        if r <= -0.999999:
            return float("nan")
        disc = 1.0
        one_plus_r = 1.0 + r
        total = -float(initial_balance)  # t=0
        tiny = float(np.finfo(float).tiny)
        for t in range(1, L):
            disc *= one_plus_r
            # Protection contre l'underflow qui pourrait rendre disc==0.0
            if not np.isfinite(disc) or abs(disc) < tiny:
                disc = tiny
            cf = 0.0
            if (t >= ss) and (t <= stp) and (t % max(1, contrib_every) == 0):
                cf = -float(
                    contrib_amount
                    * ((1 + contrib_growth_rate) ** (t // max(1, contrib_every)))
                )
            total += cf / disc
        # t=L
        if L >= 1:
            disc *= one_plus_r
            if not np.isfinite(disc) or abs(disc) < tiny:
                disc = tiny
            cf_L = -float(
                (
                    contrib_amount
                    * ((1 + contrib_growth_rate) ** (L // max(1, contrib_every)))
                    if (L >= ss) and (L <= stp) and (L % max(1, contrib_every) == 0)
                    else 0.0
                )
            ) + float(final_balance)
            total += cf_L / disc
        return total

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return np.nan
    if abs(f_lo) < tol:
        return lo
    if abs(f_hi) < tol:
        return hi
    mid = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid


def cash_sim_from_path(
    path: np.ndarray,
    steps_per_year: int,
    initial_balance: float,
    contrib_amount: float,
    contrib_every: int,
    contrib_growth_rate: float,
    start_step: int | None,
    stop_step: int | None,
    rf_annual: float,
    detailed: bool = False,
):
    L = len(path) - 1
    dtype = np.float32
    rf_step = np.float32((1 + rf_annual) ** (1 / max(1, steps_per_year)) - 1)

    if detailed:
        ret = np.ones(L + 1, dtype=dtype)
        ret[1:] = (path[1:] / path[:-1]).astype(dtype)
        contrib = np.zeros(L + 1, dtype=dtype)
        ss = start_step if start_step is not None else 1
        stp = stop_step if stop_step is not None else L
        for i in range(1, L + 1):
            if (i >= ss) and (i <= stp) and (i % max(1, contrib_every) == 0):
                contrib[i] = dtype(
                    contrib_amount
                    * ((1 + contrib_growth_rate) ** (i // max(1, contrib_every)))
                )
        balance = np.zeros(L + 1, dtype=dtype)
        balance[0] = np.float32(initial_balance)
        for i in range(1, L + 1):
            balance[i] = balance[i - 1] * ret[i] + contrib[i]
        cum_factor = path.astype(dtype)
        cum_max = np.maximum.accumulate(cum_factor)
        drawdown = np.where(cum_max > 0, cum_factor / cum_max - 1.0, 0.0).astype(dtype)
        ret_simple = (ret - 1).astype(dtype)
        ret_excess = (ret_simple - rf_step).astype(dtype)
        downside = np.minimum(0, ret_excess).astype(dtype)
        cum_contrib = np.cumsum(contrib, dtype=np.float64).astype(dtype)
        bal_max = np.maximum.accumulate(balance)
        dd_balance = np.where(bal_max > 0, balance / bal_max - 1.0, 0.0).astype(dtype)
        cash_flow = np.zeros(L + 1, dtype=np.float64)
        cash_flow[0] = -initial_balance
        if L > 1:
            cash_flow[1:L] = -contrib[1:L].astype(np.float64)
        cash_flow[L] = -float(contrib[L]) + float(balance[L])
        irr_step = bisection_irr(cash_flow)
        metrics = {
            "final_step": L,
            "final_cum_factor": float(cum_factor[-1]),
            "final_balance": float(balance[-1]),
            "total_contrib": float(cum_contrib[-1]),
            "pnl": float(balance[-1] - np.float32(initial_balance) - cum_contrib[-1]),
            "years": L / max(1, steps_per_year),
            "CAGR_returns_only": (
                float(cum_factor[-1] ** (steps_per_year / max(1, L)) - 1)
                if L > 0
                else 0.0
            ),
            "max_drawdown_returns_only": float(np.min(drawdown[1:])) if L > 0 else 0.0,
            "rf_step": float(rf_step),
            "vol_step": float(np.std(ret_simple[1:], ddof=1)) if L > 1 else 0.0,
            "vol_annual": float(
                (np.std(ret_simple[1:], ddof=1) * np.sqrt(steps_per_year))
                if L > 1
                else 0.0
            ),
            "mean_excess_step": float(np.mean(ret_excess[1:])) if L > 0 else 0.0,
            "sharpe_annual": float(
                (
                    (np.mean(ret_excess[1:]) * steps_per_year)
                    / (np.std(ret_excess[1:], ddof=1) * np.sqrt(steps_per_year) + 1e-9)
                )
                if L > 1
                else 0.0
            ),
            "sortino_annual": float(
                (
                    (np.mean(ret_excess[1:]) * steps_per_year)
                    / (np.std(downside[1:], ddof=1) * np.sqrt(steps_per_year) + 1e-9)
                )
                if L > 1
                else 0.0
            ),
            "calmar": (
                float(
                    (
                        (cum_factor[-1] ** (steps_per_year / max(1, L)) - 1)
                        / abs(np.min(drawdown[1:]))
                    )
                )
                if L > 1 and np.min(drawdown[1:]) < 0
                else np.nan
            ),
            "max_drawdown_balance": float(np.min(dd_balance[1:])) if L > 0 else 0.0,
            "equity_multiple": float(
                balance[-1]
                / max(1e-12, (np.float32(initial_balance) + cum_contrib[-1]))
            ),
            "irr_step": float(irr_step) if irr_step == irr_step else np.nan,
            "irr_annual": (
                float((1 + irr_step) ** steps_per_year - 1)
                if irr_step == irr_step
                else np.nan
            ),
        }
        df = (
            pd.DataFrame(
                {
                    "step": np.arange(L + 1, dtype=np.int32),
                    "return": ret,
                    "contrib": contrib,
                    "balance": balance,
                    "cum_factor": cum_factor,
                    "cum_max": cum_max,
                    "drawdown": drawdown,
                    "ret_simple": ret_simple,
                    "ret_excess": ret_excess,
                    "downside": downside,
                    "cum_contrib": cum_contrib,
                    "bal_max": bal_max,
                    "dd_balance": dd_balance,
                    "cash_flow": cash_flow.astype(dtype),
                }
            )
            if detailed
            else None
        )
    else:
        # Mode mémoire réduite: calculs en une passe, pas de gros tableaux
        balance = np.zeros(L + 1, dtype=dtype)
        balance[0] = np.float32(initial_balance)
        ss = start_step if start_step is not None else 1
        stp = stop_step if stop_step is not None else L
        total_contrib = 0.0
        # Stats Welford
        n = 0
        mean_rs = 0.0
        M2_rs = 0.0
        mean_ex = 0.0
        M2_ex = 0.0
        n_dn = 0
        mean_dn = 0.0
        M2_dn = 0.0
        cum_max_factor = float(path[0]) if L >= 0 else 1.0
        min_dd_returns = 0.0
        bal_max = float(balance[0])
        min_dd_balance = 0.0
        for i in range(1, L + 1):
            r_step = float(path[i] / path[i - 1] - 1.0)
            # contributions
            c_i = 0.0
            if (i >= ss) and (i <= stp) and (i % max(1, contrib_every) == 0):
                c_i = float(
                    contrib_amount
                    * ((1 + contrib_growth_rate) ** (i // max(1, contrib_every)))
                )
            total_contrib += c_i
            # balance
            balance[i] = balance[i - 1] * (1.0 + np.float32(r_step)) + np.float32(c_i)
            # stats step returns
            n += 1
            delta = r_step - mean_rs
            mean_rs += delta / n
            M2_rs += delta * (r_step - mean_rs)
            ex = r_step - float(rf_step)
            delta_ex = ex - mean_ex
            mean_ex += delta_ex / n
            M2_ex += delta_ex * (ex - mean_ex)
            if ex < 0:
                n_dn += 1
                delta_dn = ex - mean_dn
                mean_dn += delta_dn / n_dn
                M2_dn += delta_dn * (ex - mean_dn)
            # drawdowns returns only
            if float(path[i]) > cum_max_factor:
                cum_max_factor = float(path[i])
            dd_ret = float(path[i] / cum_max_factor - 1.0)
            if dd_ret < min_dd_returns:
                min_dd_returns = dd_ret
            # drawdown balance
            if float(balance[i]) > bal_max:
                bal_max = float(balance[i])
            dd_b = float(balance[i] / bal_max - 1.0)
            if dd_b < min_dd_balance:
                min_dd_balance = dd_b
        # Final metrics
        vol_step = float(np.sqrt(M2_rs / (n - 1))) if n > 1 else 0.0
        vol_annual = float(vol_step * np.sqrt(steps_per_year))
        std_ex = float(np.sqrt(M2_ex / (n - 1))) if n > 1 else 0.0
        std_dn = float(np.sqrt(M2_dn / (n_dn - 1))) if n_dn > 1 else 0.0
        sharpe_annual = (
            float(
                ((mean_ex) * steps_per_year) / (std_ex * np.sqrt(steps_per_year) + 1e-9)
            )
            if n > 1
            else 0.0
        )
        sortino_annual = (
            float(
                ((mean_ex) * steps_per_year) / (std_dn * np.sqrt(steps_per_year) + 1e-9)
            )
            if n_dn > 1
            else 0.0
        )
        geom_cagr = (
            float(path[-1] ** (steps_per_year / max(1, L)) - 1) if L > 0 else 0.0
        )
        irr_step = bisection_irr_stream(
            L=L,
            initial_balance=float(initial_balance),
            final_balance=float(balance[-1]),
            contrib_amount=float(contrib_amount),
            contrib_every=int(contrib_every),
            contrib_growth_rate=float(contrib_growth_rate),
            start_step=int(ss),
            stop_step=int(stp),
        )
        metrics = {
            "final_step": L,
            "final_cum_factor": float(path[-1]) if L >= 0 else 1.0,
            "final_balance": float(balance[-1]),
            "total_contrib": float(total_contrib),
            "pnl": float(balance[-1] - float(initial_balance) - float(total_contrib)),
            "years": L / max(1, steps_per_year),
            "CAGR_returns_only": geom_cagr,
            "max_drawdown_returns_only": float(min_dd_returns) if L > 0 else 0.0,
            "rf_step": float(rf_step),
            "vol_step": vol_step,
            "vol_annual": vol_annual,
            "mean_excess_step": float(mean_ex) if n > 0 else 0.0,
            "sharpe_annual": sharpe_annual,
            "sortino_annual": sortino_annual,
            "calmar": (
                float((geom_cagr) / abs(min_dd_returns))
                if (L > 1 and min_dd_returns < 0)
                else np.nan
            ),
            "max_drawdown_balance": float(min_dd_balance) if L > 0 else 0.0,
            "equity_multiple": float(
                balance[-1]
                / max(1e-12, (float(initial_balance) + float(total_contrib)))
            ),
            "irr_step": float(irr_step) if irr_step == irr_step else np.nan,
            "irr_annual": (
                float((1 + irr_step) ** steps_per_year - 1)
                if irr_step == irr_step
                else np.nan
            ),
        }
        df = pd.DataFrame(
            {"step": np.arange(L + 1, dtype=np.int32), "balance": balance}
        )
    return df, metrics


def create_time_series_chart(
    df, metric_col, metric_name, date_col="roundDate", ma_window=20
):
    """
    Crée un graphique temporel avec moyenne mobile pour une métrique donnée.

    Args:
        df: DataFrame contenant les données
        metric_col: nom de la colonne de la métrique
        metric_name: nom affiché pour la métrique
        date_col: nom de la colonne de date
        ma_window: fenêtre pour la moyenne mobile
    """
    if metric_col not in df.columns or date_col not in df.columns:
        return None

    # Préparer les données
    chart_df = df[[date_col, metric_col]].copy()
    chart_df = chart_df.dropna()

    if len(chart_df) == 0:
        return None

    # Trier par date
    chart_df = chart_df.sort_values(date_col)

    # Calculer la moyenne mobile
    chart_df[f"{metric_col}_ma"] = (
        chart_df[metric_col].rolling(window=ma_window, min_periods=1).mean()
    )

    # Renommer les colonnes pour l'affichage
    chart_df = chart_df.rename(
        columns={
            metric_col: f"{metric_name}",
            f"{metric_col}_ma": f"{metric_name} (MA{ma_window})",
        }
    )

    # Reshape pour Altair
    melted_df = pd.melt(
        chart_df,
        id_vars=[date_col],
        value_vars=[f"{metric_name}", f"{metric_name} (MA{ma_window})"],
        var_name="Série",
        value_name="Valeur",
    )

    # Créer le graphique
    chart = (
        alt.Chart(melted_df)
        .mark_line()
        .encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("Valeur:Q", title=metric_name),
            color=alt.Color(
                "Série:N",
                scale=alt.Scale(
                    domain=[f"{metric_name}", f"{metric_name} (MA{ma_window})"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date"),
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("Valeur:Q", title="Valeur", format=".4f"),
            ],
        )
        .properties(
            width=700, height=300, title=f"Évolution temporelle - {metric_name}"
        )
        .resolve_scale(color="independent")
    )

    # Ajouter une ligne de référence à 0 si approprié
    if metric_name in ["CORRv2", "MMC"]:
        ref_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y:Q")
        )
        chart = chart + ref_line
    elif metric_name == "Season Score Payout":
        ref_line = (
            alt.Chart(pd.DataFrame({"y": [1]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y:Q")
        )
        chart = chart + ref_line

    return chart


@st.cache_data(show_spinner=False, ttl=300)
def get_models_cached(pub, sec):
    return NumerAPI(public_id=pub, secret_key=sec).get_models()


@st.cache_data(show_spinner=True, ttl=300, max_entries=32)
def fetch_v2_rounds(model_id, pub, sec):
    api = NumerAPI(public_id=pub, secret_key=sec)
    res = api.raw_query(QUERY_V2, {"modelId": model_id})
    return res["data"]["v2RoundModelPerformances"]


with st.sidebar:
    st.header("Source")
    source = st.radio("Données", ["API", "Upload JSON"], index=0)
    resolved_only = st.checkbox("Rounds résolus uniquement", value=False)
    st.header("Paramètres de simulation")
    L = st.number_input("Longueur L", 10, 10000, 750, 10)
    n_paths = st.number_input("n_paths", 1, 5000, 100, 10)
    n_paths_plot = st.slider(
        "Trajectoires à afficher", 1, min(100, int(n_paths)), min(20, int(n_paths))
    )
    plot_every = st.number_input("Décimation tracé (chaque k pas)", 1, 1000, 1, 1)
    horizons_txt = st.text_input(
        "Horizons (csv)", value="4,8,12,20,40,60,120,250,375,500,750,1000,1250,1500"
    )
    n_sims = st.number_input("Sims / horizon", 100, 200000, 100, 100)
    low_mem_stats = st.checkbox(
        "Stats terminales: mode mémoire réduite (approx)", value=True
    )
    steps_per_year = st.number_input(
        "Pas/an (21 steps ~ 1 Mois, 251 ~ 1 Ans)", 1, 2000, 251
    )
    initial_balance = st.number_input("Solde initial", 0.0, value=1000.0, step=100.0)
    contrib_amount = st.number_input("Apport", 0.0, value=100.0, step=10.0)
    contrib_every = st.number_input(
        "Apport tous les n pas (21 steps ~ 1 Mois)", 1, 3650, 20
    )
    selected_path = st.number_input(
        "Trajectoire sélectionnée (1..n_paths)", 1, int(n_paths), 1
    )
    st.divider()
    rf_annual = st.number_input(
        "Taux RF annualisé", -1.0, 5.0, 0.0, 0.01, format="%.4f"
    )
    contrib_growth_rate = st.number_input(
        "Croissance apports/période", -1.0, 5.0, 0.0, 0.01, format="%.4f"
    )
    contrib_start_step = st.number_input("Début apports (step)", 1, 100000, 1)
    contrib_stop_step = st.text_input("Fin apports (step, vide = L)", value="")
    st.divider()
    max_rows_show = st.slider("Lignes à afficher (tables)", 50, 5000, 1000, 50)

if "rounds" not in st.session_state:
    st.session_state["rounds"] = None
if "current_model" not in st.session_state:
    st.session_state["current_model"] = None
all_rounds = []
errors = []

if source == "Upload JSON":
    uploaded = st.file_uploader(
        "Déposez des JSON Numerai (v2RoundModelPerformances)",
        type=["json"],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("Déposez des fichiers ou passez sur API.")
        st.stop()
    for f in uploaded:
        try:
            data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
            all_rounds.extend(_load_rounds_from_obj(data))
        except Exception as e:
            errors.append(f"{getattr(f,'name','in-mem')}: {e}")
    st.session_state["rounds"] = all_rounds
    st.session_state["current_model"] = "__upload__"
else:
    st.subheader("Connexion API Numerai")
    public_id = st.secrets.get("NUMERAI_PUBLIC_ID", "")
    secret_key = st.secrets.get("NUMERAI_SECRET_KEY", "")
    if not public_id or not secret_key:
        st.info("Renseigne tes clés pour lister les modèles.")
        st.stop()
    try:
        models_map = get_models_cached(public_id, secret_key)
        if not models_map:
            st.error("Aucun modèle trouvé.")
            st.stop()
        model_name = st.selectbox("Modèle", sorted(models_map.keys()))
        c1, c2 = st.columns(2)
        refresh = c1.button("Charger / Refresh")
        keep_cached = c2.checkbox("Utiliser le cache", value=True)
        need_fetch = (
            refresh
            or st.session_state["rounds"] is None
            or st.session_state["current_model"] != model_name
        )
        if need_fetch:
            with st.spinner("Récupération des rounds..."):
                if not keep_cached:
                    fetch_v2_rounds.clear()
                rounds = fetch_v2_rounds(models_map[model_name], public_id, secret_key)
            st.session_state["rounds"] = rounds
            st.session_state["current_model"] = model_name
        all_rounds = st.session_state["rounds"] or []
    except Exception as e:
        st.error(f"Erreur API Numerai: {e}")
        st.stop()

if errors:
    with st.expander("Erreurs de parsing", expanded=False):
        for e in errors:
            st.error(e)

if not (st.session_state["rounds"] or all_rounds):
    st.warning("Aucun round valide détecté.")
    st.stop()

rounds_in = st.session_state["rounds"] if source == "API" else all_rounds
long_df = build_long_df(rounds_in)

# Filtre optionnel: ne garder que les rounds dont roundResolveTime est passé
if resolved_only:
    now = pd.Timestamp.utcnow().tz_localize(None)
    if "roundResolveTime" in long_df.columns:
        long_df = long_df[long_df["roundResolveTime"].notna()]
        long_df = long_df[long_df["roundResolveTime"] <= now]

mask = (
    long_df["displayName"].astype(str).str.contains("mmc|corr", case=False, regex=True)
    & long_df["value"].notna()
)
keep = sorted(long_df.loc[mask, "roundNumber"].dropna().unique().tolist())
base = long_df[long_df["roundNumber"].isin(keep)].copy()
# Réduire la largeur: ne garder que mmc|corr|season pour les pivots
try:
    base = base[
        base["displayName"]
        .astype(str)
        .str.contains("mmc|corr|season", case=False, regex=True)
    ].copy()
except Exception:
    pass

index_cols = [
    "roundNumber",
    "roundDate",
    "roundScoreTime",
    "roundResolveTime",
    "roundPayoutFactor",
    "atRisk",
]
# Tri + déduplication pour garder la première observation par (index_cols, displayName)
_tmp = base.sort_values(index_cols + ["displayName"]).drop_duplicates(
    index_cols + ["displayName"], keep="first"
)
# Pivot sans agrégation (évite le produit cartésien)
values_df = _tmp.pivot(
    index=index_cols, columns="displayName", values="value"
).reset_index()
percentiles_df = _tmp.pivot(
    index=index_cols, columns="displayName", values="percentile"
).reset_index()
for c in values_df.columns:
    if c in ("roundPayoutFactor", "atRisk"):
        values_df[c] = pd.to_numeric(values_df[c], errors="coerce").astype("float32")
    elif c == "roundNumber":
        values_df[c] = pd.to_numeric(values_df[c], errors="coerce", downcast="integer")
    elif c not in index_cols and pd.api.types.is_float_dtype(values_df[c]):
        values_df[c] = values_df[c].astype("float32")

percentiles_df = percentiles_df[
    index_cols + sorted([c for c in percentiles_df.columns if c not in index_cols])
]
for c in percentiles_df.columns:
    if c in ("roundPayoutFactor", "atRisk"):
        percentiles_df[c] = pd.to_numeric(percentiles_df[c], errors="coerce").astype(
            "float32"
        )
    elif c == "roundNumber":
        percentiles_df[c] = pd.to_numeric(
            percentiles_df[c], errors="coerce", downcast="integer"
        )
    elif c not in index_cols and pd.api.types.is_float_dtype(percentiles_df[c]):
        percentiles_df[c] = percentiles_df[c].astype("float32")

season_col = find_season_col(values_df.columns)
if (season_col is not None) and ("roundPayoutFactor" in values_df.columns):
    values_df["season_score_payout"] = 1 + (
        pd.to_numeric(values_df[season_col], errors="coerce")
        * pd.to_numeric(values_df["roundPayoutFactor"], errors="coerce")
    )

st.success(
    f"✅ **Analyse terminée** - Rounds analysés: {len(keep)} | "
    f"Période: {values_df['roundDate'].min().strftime('%Y-%m-%d') if pd.notna(values_df['roundDate'].min()) else 'N/A'} → "
    f"{values_df['roundDate'].max().strftime('%Y-%m-%d') if pd.notna(values_df['roundDate'].max()) else 'N/A'}"
)

T5, T1, T2, T6, T3, T4 = st.tabs(
    [
        "📊 Résumé",
        "📋 Données",
        "📈 Distribution",
        "📈 Évolution Temporelle",
        "🎲 Simulations",
        "💰 Cash sim",
    ]
)

with T1:
    st.subheader("Values")
    st.dataframe(values_df.head(int(max_rows_show)), use_container_width=True)
    st.subheader("Percentiles")
    st.dataframe(percentiles_df.head(int(max_rows_show)), use_container_width=True)
    st.subheader("Long (raw)")
    st.dataframe(long_df.head(int(max_rows_show)), use_container_width=True)

# Libération mémoire
try:
    del base
except Exception:
    pass
gc.collect()

with T2:
    st.subheader("Distribution de season_score_payout")
    if "season_score_payout" in values_df.columns:
        ssp = pd.to_numeric(values_df["season_score_payout"], errors="coerce")
        stats = pctile_stats(ssp)
        hist_df = make_hist(ssp, bins=20)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("Statistiques")
            st.dataframe(stats, use_container_width=True)
        with c2:
            st.markdown("Histogramme")
            if len(hist_df) > 0:
                # Définir un domaine X qui inclut 0 pour afficher une règle verticale à x=0
                x_min = float(hist_df["bin_left"].min())
                x_max = float(hist_df["bin_right"].max())
                dom_min = min(x_min, 1.0)
                dom_max = max(x_max, 1.0)

                hist_chart = (
                    alt.Chart(hist_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "bin_mid:Q",
                            title="season_score_payout",
                            scale=alt.Scale(domain=[dom_min, dom_max]),
                        ),
                        y=alt.Y("count:Q", title="Nombre"),
                        tooltip=[
                            alt.Tooltip("bin_left:Q", title="Bin gauche"),
                            alt.Tooltip("bin_right:Q", title="Bin droite"),
                            alt.Tooltip("count:Q", title="Compte"),
                        ],
                    )
                )
                vline = (
                    alt.Chart(pd.DataFrame({"x": [1.0]}))
                    .mark_rule(color="red")
                    .encode(x="x:Q")
                )
                st.altair_chart(hist_chart + vline, use_container_width=True)

                # Courbe de densité (KDE) basée sur la série brute
                ssp_clean = ssp.dropna().astype(float)
                if ssp_clean.size > 1:
                    df_kde = pd.DataFrame({"ssp": ssp_clean.to_numpy()})
                    kde_chart = (
                        alt.Chart(df_kde)
                        .transform_density(
                            "ssp",
                            as_=["ssp", "density"],
                            extent=[float(ssp_clean.min()), float(ssp_clean.max())],
                            steps=200,
                        )
                        .mark_line(color="orange")
                        .encode(
                            x=alt.X("ssp:Q", title="season_score_payout"),
                            y=alt.Y("density:Q", title="Densité"),
                        )
                    )
                    st.altair_chart(kde_chart, use_container_width=True)

                    # Panneau d'échantillonnage depuis la distribution empirique
                    with st.expander("Échantillonnage empirique"):
                        c3, c4 = st.columns(2)
                        sample_n = int(
                            c3.number_input(
                                "Taille de l'échantillon", 10, 100000, 1000, 100
                            )
                        )
                        sample_seed = int(
                            c4.number_input("Seed (sample)", 0, 2**32 - 1, 42)
                        )
                        rng_s = np.random.default_rng(sample_seed)
                        sample = rng_s.choice(
                            ssp_clean.to_numpy(), size=sample_n, replace=True
                        )
                        # Mini-histogramme de l'échantillon
                        shist_df = make_hist(pd.Series(sample), bins=20)
                        schart = (
                            alt.Chart(shist_df)
                            .mark_bar(color="#4C78A8")
                            .encode(
                                x=alt.X("bin_mid:Q", title="Échantillon (ssp)"),
                                y=alt.Y("count:Q", title="Nombre"),
                                tooltip=[
                                    alt.Tooltip("bin_left:Q", title="Bin gauche"),
                                    alt.Tooltip("bin_right:Q", title="Bin droite"),
                                    alt.Tooltip("count:Q", title="Compte"),
                                ],
                            )
                        )
                        st.altair_chart(schart, use_container_width=True)
                        # Stats rapides
                        c5, c6, c7 = st.columns(3)
                        c5.metric("Moyenne", f"{float(np.mean(sample)):.5f}")
                        c6.metric("Médiane", f"{float(np.median(sample)):.5f}")
                        c7.metric("Écart-type", f"{float(np.std(sample, ddof=1)):.5f}")
            else:
                st.info("Pas de données pour l'histogramme.")

with T6:
    st.subheader("Évolution temporelle des métriques")

    # Paramètres de la moyenne mobile
    ma_window = st.slider("Fenêtre moyenne mobile (jours)", 5, 100, 20, 5)

    # Vérifier quelles colonnes sont disponibles
    available_metrics = []
    metric_mapping = {}

    # Season Score Payout
    if "season_score_payout" in values_df.columns:
        available_metrics.append("Season Score Payout")
        metric_mapping["Season Score Payout"] = "season_score_payout"

    # Rechercher CORRv2
    corr_cols = [
        col
        for col in values_df.columns
        if "corr" in str(col).lower() and "v2" in str(col).lower()
    ]
    if corr_cols:
        available_metrics.append("CORRv2")
        metric_mapping["CORRv2"] = corr_cols[0]

    # Rechercher MMC
    mmc_cols = [col for col in values_df.columns if "mmc" in str(col).lower()]
    if mmc_cols:
        available_metrics.append("MMC")
        metric_mapping["MMC"] = mmc_cols[0]

    if not available_metrics:
        st.warning("Aucune métrique disponible pour les graphiques temporels.")
    else:
        st.info(f"Métriques disponibles: {', '.join(available_metrics)}")

        # Créer les graphiques pour chaque métrique disponible
        for metric_name in available_metrics:
            metric_col = metric_mapping[metric_name]

            st.markdown(f"### {metric_name}")

            # Créer le graphique
            chart = create_time_series_chart(
                values_df,
                metric_col,
                metric_name,
                date_col="roundDate",
                ma_window=ma_window,
            )

            if chart is not None:
                st.altair_chart(chart, use_container_width=True)

                # Statistiques rapides
                metric_data = pd.to_numeric(
                    values_df[metric_col], errors="coerce"
                ).dropna()
                if len(metric_data) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Moyenne", f"{metric_data.mean():.4f}")
                    with col2:
                        st.metric("Médiane", f"{metric_data.median():.4f}")
                    with col3:
                        st.metric("Écart-type", f"{metric_data.std():.4f}")
                    with col4:
                        # Calculer la moyenne mobile récente
                        recent_data = values_df.sort_values("roundDate").tail(
                            ma_window
                        )[metric_col]
                        recent_ma = pd.to_numeric(recent_data, errors="coerce").mean()
                        if not pd.isna(recent_ma):
                            st.metric(f"MA{ma_window} récente", f"{recent_ma:.4f}")
                        else:
                            st.metric(f"MA{ma_window} récente", "N/A")
            else:
                st.warning(f"Impossible de créer le graphique pour {metric_name}")

            st.divider()

with T3:
    st.subheader("Trajectoires simulées et stats terminales")
    ssp = get_ssp_series(values_df).to_numpy(dtype=float)
    ssp = ssp[~np.isnan(ssp)]
    if ssp.size == 0:
        st.warning(
            "Aucune donnée de season_score_payout pour simuler des trajectoires."
        )
    else:
        try:
            horizons = [
                int(x.strip()) for x in horizons_txt.split(",") if x.strip().isdigit()
            ]
            if not horizons:  # Si la liste est vide après parsing
                horizons = [4, 8, 12, 20, 40, 60, 120, 250, 375, 500, 750]
        except Exception:
            horizons = [4, 8, 12, 20, 40, 60, 120, 250, 375, 500, 750]
        seed = st.number_input("Seed RNG", 0, 2**32 - 1, 0)
        rng = np.random.default_rng(int(seed))
        with st.spinner("Simulation de trajectoires..."):
            n_paths_eff = int(min(int(n_paths), int(n_paths_plot)))
            paths_df = simulate_paths(ssp, int(L), n_paths_eff, rng)
        st.caption(f"Trajectoires générées: {paths_df.shape[1]-1} | Longueur: {int(L)}")

        # Préparer les données pour le graphique
        plot_cols = ["step"] + [c for c in paths_df.columns[1 : 1 + int(n_paths_plot)]]
        plot_df = paths_df[plot_cols].copy()
        if int(plot_every) > 1:
            plot_df = plot_df.iloc[:: int(plot_every), :]

        # Afficher le graphique
        st.line_chart(plot_df.set_index("step"))

        # Nettoyage mémoire
        del paths_df, plot_df
        gc.collect()
        with st.spinner("Stats terminales par horizon..."):
            if low_mem_stats:
                hstats_df = terminal_stats_stream(ssp, horizons, int(n_sims), rng)
            else:
                hstats_df = terminal_stats(ssp, horizons, int(n_sims), rng)
        st.dataframe(hstats_df, use_container_width=True)

with T4:
    st.subheader("Simulation de solde avec apports")
    ssp2 = get_ssp_series(values_df).to_numpy(dtype=float)
    ssp2 = ssp2[~np.isnan(ssp2)]
    if ssp2.size == 0:
        st.warning("Aucune donnée de season_score_payout pour la simulation de cash.")
    else:
        try:
            horizons = [
                int(x.strip()) for x in horizons_txt.split(",") if x.strip().isdigit()
            ]
            if not horizons:  # Si la liste est vide après parsing
                horizons = [4, 8, 12, 20, 40, 60, 120, 250, 375, 500, 750]
        except Exception:
            horizons = [4, 8, 12, 20, 40, 60, 120, 250, 375, 500, 750]
        seed2 = st.number_input("Seed RNG (cash sim)", 0, 2**32 - 1, 1)
        rng2 = np.random.default_rng(int(seed2))
        show_details = st.checkbox("Afficher les détails complets", value=False)
        sel_idx = max(1, min(int(n_paths), int(selected_path)))
        # Génère uniquement la trajectoire sélectionnée pour économiser la mémoire
        rng_path = np.random.default_rng(int(seed2) + int(sel_idx) - 1)
        path = simulate_single_path(ssp2, int(L), rng_path)
        try:
            stop_val = (
                int(contrib_stop_step) if contrib_stop_step.strip() != "" else int(L)
            )
        except Exception:
            stop_val = int(L)
        sim_df, metrics = cash_sim_from_path(
            path=path,
            steps_per_year=int(steps_per_year),
            initial_balance=float(initial_balance),
            contrib_amount=float(contrib_amount),
            contrib_every=int(contrib_every),
            contrib_growth_rate=float(contrib_growth_rate),
            start_step=int(contrib_start_step),
            stop_step=int(stop_val),
            rf_annual=float(rf_annual),
            detailed=show_details,
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            if (
                sim_df is not None
                and "step" in sim_df.columns
                and "balance" in sim_df.columns
            ):
                plot_df = sim_df[["step", "balance"]]
                if int(plot_every) > 1:
                    plot_df = plot_df.iloc[:: int(plot_every), :]
                st.line_chart(plot_df.set_index("step")["balance"], height=300)
            else:
                st.info("Aucune donnée de solde à tracer.")
        with c2:
            st.markdown("Métriques")
            st.dataframe(
                pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
            )
        with st.expander("Données détaillées (cash sim)"):
            if sim_df is not None and show_details:
                st.dataframe(sim_df.head(int(max_rows_show)), use_container_width=True)
            elif sim_df is not None:
                st.info(
                    "Activez 'Afficher les détails complets' pour voir le DataFrame détaillé."
                )
            else:
                st.info("Aucune donnée détaillée disponible.")
        # Libération mémoire
        try:
            del path
        except Exception:
            pass
        gc.collect()

with T5:
    st.subheader("Résumé")
    ssp_series = get_ssp_series(values_df)
    ssp = ssp_series.to_numpy(dtype=float)
    ssp = ssp[~np.isnan(ssp)]
    if ssp.size == 0:
        st.info("Aucune donnée season_score_payout disponible.")
    else:
        ssp_pos = ssp[ssp > 0]
        dropped = int(ssp.size - ssp_pos.size)
        if ssp_pos.size == 0:
            st.warning("Impossible de calculer le log: toutes les valeurs sont <= 0.")
        else:
            levels = [0.80, 0.90, 0.95, 0.975, 0.99]
            labels = ["80%", "90%", "95%", "97.5%", "99%"]
            idx_default = 2
            sel_label = st.selectbox("Niveau de confiance", labels, index=idx_default)
            if sel_label is None:
                sel_label = labels[idx_default]
            level = levels[labels.index(sel_label)]
            z_map = {
                0.80: 1.2816,
                0.90: 1.6449,
                0.95: 1.96,
                0.975: 2.2414,
                0.99: 2.5758,
            }
            z = z_map.get(level, 1.96)
            logs = np.log(ssp_pos)
            n = logs.size
            mean_log = float(np.mean(logs))
            std_log = float(np.std(logs, ddof=1)) if n > 1 else 0.0
            se_log = float(std_log / np.sqrt(n)) if n > 1 else float("nan")
            geom_factor = float(np.exp(mean_log))
            geom_rate = geom_factor - 1.0
            ci_low_rate = (
                float(np.exp(mean_log - z * se_log) - 1.0) if n > 1 else float("nan")
            )
            ci_high_rate = (
                float(np.exp(mean_log + z * se_log) - 1.0) if n > 1 else float("nan")
            )
            se_rate_delta = float(geom_factor * se_log) if n > 1 else float("nan")
            ann_rate = float((1 + geom_rate) ** int(steps_per_year) - 1.0)
            start_d = (
                pd.to_datetime(values_df["roundDate"].min())
                if "roundDate" in values_df.columns
                else None
            )
            end_d = (
                pd.to_datetime(values_df["roundDate"].max())
                if "roundDate" in values_df.columns
                else None
            )
            c1, c2 = st.columns([2, 1])
            with c1:
                st.metric("Taux moyen par jour (géométrique)", f"{geom_rate*100:.3f}%")
                st.metric("Taux annualisé (géométrique)", f"{ann_rate*100:.2f}%")
            with c2:
                st.write(
                    f"Observations (utilisées/total): {n}/{ssp.size} (écartées: {dropped})"
                )
                if start_d is not None and end_d is not None:
                    st.write(f"Période: {start_d.date()} → {end_d.date()}")
            st.markdown("Détails statistiques")
            summary = pd.DataFrame(
                [
                    {"metric": "mean_log", "value": mean_log},
                    {"metric": "std_log", "value": std_log},
                    {"metric": "se_log", "value": se_log},
                    {"metric": "geom_factor", "value": geom_factor},
                    {"metric": "geom_rate", "value": geom_rate},
                    {"metric": f"CI {sel_label} bas (taux)", "value": ci_low_rate},
                    {"metric": f"CI {sel_label} haut (taux)", "value": ci_high_rate},
                    {"metric": "SE (taux, delta)", "value": se_rate_delta},
                    {"metric": "n", "value": n},
                    {"metric": "écartés (<=0)", "value": dropped},
                ]
            )
            st.dataframe(summary, use_container_width=True)
            st.caption(
                "SE (taux, delta) ≈ exp(mean_log) × SE(mean_log). C’est l’erreur standard du taux géométrique par pas, obtenue par la méthode delta."
            )
            st.markdown("Projections (géométrique)")
            years = st.slider("Horizon (années)", min_value=1, max_value=20, value=5)
            H = int(years * int(steps_per_year))
            if n > 1 and std_log > 0:
                if not np.isnan(se_log):
                    mu_low = mean_log - z * se_log
                    mu_high = mean_log + z * se_log
                    F_param_low = float(np.exp(H * mu_low))
                    F_param_high = float(np.exp(H * mu_high))
                    r_ann_param_low = float(F_param_low ** (1.0 / years) - 1.0)
                    r_ann_param_high = float(F_param_high ** (1.0 / years) - 1.0)
                else:
                    F_param_low = F_param_high = np.nan
                    r_ann_param_low = r_ann_param_high = np.nan
                mu_sum = H * mean_log
                sigma_sum = np.sqrt(H) * std_log
                F_pred_median = float(np.exp(mu_sum))
                F_pred_low = float(np.exp(mu_sum - z * sigma_sum))
                F_pred_high = float(np.exp(mu_sum + z * sigma_sum))
                r_ann_pred_low = float(F_pred_low ** (1.0 / years) - 1.0)
                r_ann_pred_high = float(F_pred_high ** (1.0 / years) - 1.0)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("IC paramétrique (sur la moyenne)")
                    st.write(
                        f"Facteur cumulé {years} ans: [{F_param_low:.3f} ; {F_param_high:.3f}]"
                    )
                    st.write(
                        f"Taux annualisé: [{r_ann_param_low*100:.2f}% ; {r_ann_param_high*100:.2f}%]"
                    )
                with c2:
                    st.markdown("Intervalle prédictif (aléa futur)")
                    st.write(
                        f"Médiane du facteur cumulé {years} ans: {F_pred_median:.3f}"
                    )
                    st.write(
                        f"PI facteur cumulé {years} ans: [{F_pred_low:.3f} ; {F_pred_high:.3f}]"
                    )
                    st.write(
                        f"PI taux annualisé: [{r_ann_pred_low*100:.2f}% ; {r_ann_pred_high*100:.2f}%]"
                    )
            else:
                st.info(
                    "Échantillon insuffisant pour des projections (écart-type nul ou n ≤ 1)."
                )

st.caption(
    "💡 **Conseils d'utilisation :** Import via JSON ou directement depuis l'API Numerai. "
    "Les appels API restent en cache tant que vous ne cliquez pas sur Refresh. "
    "Pour les simulations longues, utilisez un nombre de trajectoires raisonnable."
)
