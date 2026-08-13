"""Daily learning report helpers."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.config import ROOT, LearningConfig, SessionBucketConfig
from trading_system.database import Database
from trading_system.learning import LearningEngine, daily_objective_progress, trade_session
from trading_system.learning.sessions import (
    DEFAULT_SESSION_BUCKETS,
    pattern_session_name,
    session_hours_label,
)
from trading_system.types import Position


def _pattern_decision_block(
    patterns: list[dict[str, Any]],
    *,
    threshold: int,
    direction: str,
    confirmed: bool,
    limit: int = 20,
) -> list[str]:
    lines: list[str] = []
    ordered = sorted(patterns, key=lambda x: -int(x.get("count") or 0))[:limit]
    if not ordered:
        lines.append("- Ninguno.")
        return lines

    for p in ordered:
        key = p["pattern_key"]
        count = int(p.get("count") or 0)
        conf_count = p.get("confirmed_count")
        conf_at = p.get("confirmed_at") or "—"
        effect = p.get("effect_action") or "—"
        reason = p.get("decision_reason")

        if confirmed:
            reps = conf_count if conf_count is not None else count
            if not reason:
                reason = (
                    f"ACEPTADO: '{key}' alcanzó {reps} repeticiones "
                    f"(umbral ≥{threshold})."
                )
            lines.append(f"- **`{key}`**")
            lines.append(f"  - Decisión: **ACEPTADO / CONFIRMADO** ({direction})")
            lines.append(f"  - Repeticiones al confirmar: **{reps}** (umbral ≥{threshold})")
            lines.append(f"  - Count actual: **{count}** | Confirmado en: {conf_at}")
            lines.append(f"  - Efecto: `{effect}`")
            lines.append(f"  - Razón: {reason}")
        else:
            remaining = max(0, threshold - count)
            reason = (
                f"RECHAZADO como patrón confirmado (aún en observación): '{key}' "
                f"solo tiene {count}/{threshold} repeticiones; faltan {remaining} "
                f"para aceptarlo. No se aplica boost/penalty/soft-reject todavía."
            )
            lines.append(f"- **`{key}`**")
            lines.append(f"  - Decisión: **NO ACEPTADO / OBSERVANDO** ({direction})")
            lines.append(f"  - Repeticiones: **{count}/{threshold}** (faltan {remaining})")
            lines.append(f"  - Efecto: ninguno todavía")
            lines.append(f"  - Razón: {reason}")
    return lines


def _session_buckets(learning: LearningEngine | None) -> list[SessionBucketConfig]:
    if learning is not None and learning.cfg.session_buckets:
        return list(learning.cfg.session_buckets)
    return list(DEFAULT_SESSION_BUCKETS)


def _learning_cfg(learning: LearningEngine | None) -> LearningConfig:
    if learning is not None:
        return learning.cfg
    return LearningConfig()


def _session_trade_stats(
    trades: list[Position],
    cfg: LearningConfig,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by: dict[str, list[Position]] = defaultdict(list)
    for t in trades:
        by[trade_session(t, cfg)].append(t)

    for name, items in by.items():
        n = len(items)
        wins = [t for t in items if (t.pnl or 0) > 0]
        losses = [t for t in items if (t.pnl or 0) <= 0]
        tp = sum(1 for t in items if t.exit_reason == "take_profit")
        sl = sum(1 for t in items if t.exit_reason == "stop_loss")
        ts = sum(1 for t in items if t.exit_reason == "time_stop")
        liq = sum(1 for t in items if t.exit_reason == "liquidation")
        pnl = sum(t.pnl or 0 for t in items)
        wr = len(wins) / n if n else 0.0
        out[name] = {
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": wr,
            "pnl": pnl,
            "take_profit": tp,
            "stop_loss": sl,
            "time_stop": ts,
            "liquidation": liq,
        }
    return out


def _filter_patterns_for_session(
    patterns: list[dict[str, Any]], session: str
) -> list[dict[str, Any]]:
    prefix = f"session={session}"
    return [
        p
        for p in patterns
        if p.get("pattern_key", "").startswith(prefix + "|")
        or p.get("pattern_key") == prefix
    ]


def _filter_changes_for_session(
    changes: list[dict[str, Any]], session: str
) -> list[dict[str, Any]]:
    prefix = f"session={session}"
    return [
        c
        for c in changes
        if str(c.get("pattern_key", "")).startswith(prefix + "|")
        or str(c.get("pattern_key", "")) == prefix
    ]


def _session_report_section(
    *,
    buckets: list[SessionBucketConfig],
    day_trades: list[Position],
    cfg: LearningConfig,
    win_patterns: list[dict[str, Any]],
    loss_patterns: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    threshold: int,
) -> list[str]:
    lines: list[str] = [
        "",
        "## 6. Resumen por horario / región (UTC)",
        "",
        "Cada bloque usa `entry_time` del trade para asignar sesión. "
        "Patrones y cambios listados son los scoped a esa sesión "
        "(`session=<name>|…`).",
        "",
    ]
    stats = _session_trade_stats(day_trades, cfg)
    confirmed_wins = [p for p in win_patterns if p.get("status") == "confirmed"]
    confirmed_losses = [p for p in loss_patterns if p.get("status") == "confirmed"]

    for b in buckets:
        name = b.name
        hours = session_hours_label(b)
        st = stats.get(
            name,
            {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "pnl": 0.0,
                "take_profit": 0,
                "stop_loss": 0,
                "time_stop": 0,
                "liquidation": 0,
            },
        )
        lines += [
            f"### `{name}` ({hours})",
            f"- Trades: **{st['trades']}** (W {st['wins']} / L {st['losses']})",
            f"- Win rate: **{st['win_rate']:.1%}** | PnL sesión: **{st['pnl']:.4f}**",
            f"- Exits: take_profit=**{st['take_profit']}** | stop_loss=**{st['stop_loss']}** "
            f"| time_stop=**{st['time_stop']}** | liquidation=**{st['liquidation']}**",
            "",
            "#### Patrones confirmados de ganancia",
        ]
        sess_wins = _filter_patterns_for_session(confirmed_wins, name)
        lines += _pattern_decision_block(
            sess_wins, threshold=threshold, direction="win", confirmed=True, limit=10
        )
        lines += ["", "#### Patrones confirmados de pérdida"]
        sess_losses = _filter_patterns_for_session(confirmed_losses, name)
        lines += _pattern_decision_block(
            sess_losses, threshold=threshold, direction="loss", confirmed=True, limit=10
        )
        lines += ["", "#### Cambios / efectos aplicados (estrategia·indicador)"]
        sess_changes = _filter_changes_for_session(changes, name)
        if sess_changes:
            for c in sess_changes:
                lines.append(
                    f"- [{c['action']}] `{c['pattern_key']}` ({c['direction']}): {c['detail']}"
                )
        else:
            lines.append("- Ninguno en esta sesión hoy.")
        lines.append("")
    return lines


def write_daily_report(
    db: Database,
    learning: LearningEngine | None = None,
    out_dir: Path | None = None,
    day: str | None = None,
) -> Path:
    out_dir = out_dir or (ROOT / "reports" / "daily")
    out_dir.mkdir(parents=True, exist_ok=True)
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"daily_{day}.md"

    day_trades = db.closed_trades_on_day(day)
    all_closed = db.get_all_closed()
    rankings = db.get_strategy_stats()
    day_insights = db.insights_on_day(day)
    changes = db.applied_changes_on_day(day)
    win_patterns = db.get_patterns(direction="win")
    loss_patterns = db.get_patterns(direction="loss")
    rejected = db.recent_rejected(50)
    cfg = _learning_cfg(learning)
    buckets = _session_buckets(learning)

    wins = [t for t in day_trades if (t.pnl or 0) > 0]
    losses = [t for t in day_trades if (t.pnl or 0) <= 0]
    day_pnl = sum(t.pnl or 0 for t in day_trades)
    wr = len(wins) / len(day_trades) if day_trades else 0.0
    exp = day_pnl / len(day_trades) if day_trades else 0.0

    confirmed_wins = [p for p in win_patterns if p["status"] == "confirmed"]
    confirmed_losses = [p for p in loss_patterns if p["status"] == "confirmed"]
    observing_wins = [p for p in win_patterns if p["status"] == "observing"]
    observing_losses = [p for p in loss_patterns if p["status"] == "observing"]

    threshold = cfg.pattern_min_occurrences
    if learning is not None:
        progress = learning.phase_progress(len(all_closed))
    else:
        progress = {
            "suggested_phase": "unknown",
            "exploration_ratio": 0,
            "confirmed_wins": len(confirmed_wins),
            "confirmed_losses": len(confirmed_losses),
            "session_aware": cfg.session_aware,
            "current_session": "—",
        }

    resets = int(db.get_state("capital_resets") or "0")
    cash = db.get_state("cash") or "?"

    start_eq = db.first_equity_on_day(day)
    end_eq = None
    try:
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT equity FROM equity_curve
                WHERE substr(timestamp, 1, 10) = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (day,),
            ).fetchone()
        if row:
            end_eq = float(row["equity"])
    except Exception:
        end_eq = None
    obj = daily_objective_progress(
        start_equity=start_eq,
        current_equity=end_eq if end_eq is not None else (start_eq or 0.0),
        target_pct=50.0,
        phase=str(progress.get("suggested_phase") or "discovery"),
        chase_in_discovery=False,
    )

    tp_n = sum(1 for t in day_trades if t.exit_reason == "take_profit")
    sl_n = sum(1 for t in day_trades if t.exit_reason == "stop_loss")
    te_n = sum(1 for t in day_trades if t.exit_reason == "trend_exit")
    ts_n = sum(1 for t in day_trades if t.exit_reason == "time_stop")

    lines = [
        f"# Daily Learning Report — {day}",
        "",
        "## 0. Ajustes activos (contexto reciente)",
        "- Session-aware learning: patrones win/loss por bucket UTC "
        f"({'ON' if cfg.session_aware else 'OFF'})",
        "- Stop-loss fee-aware: budget = pérdida neta máx. (incluye fee de salida); TP sin cambio",
        "- Take-profit: `trend_fade` (no fixed TP); SL = 4% margin NET",
        "- No time_stop: hold until fade / stop_loss / liquidation / FX session_end",
        "- Rejection candle obligatoria + no entrar si ya se alejó demasiado del extremo",
        "- Paper leverage/margin según config (fees/PnL sobre notional)",
        "- Objetivo: maximizar equity; norte +50%/día UTC en explotación (discovery no persigue el target)",
        "- Monitor: entry/exit px + etiqueta aprendizaje ganancia/pérdida; banner de sesión",
        "",
        "## 1. Aprendizaje del día",
        f"- Closed trades today: **{len(day_trades)}** (W {len(wins)} / L {len(losses)})",
        f"- Win rate today: **{wr:.1%}** | Expectancy: **{exp:.4f}** | Day PnL: **{day_pnl:.4f}**",
        f"- Equity UTC day: start=**{obj['day_start_equity']:.2f}** "
        f"end=**{obj['current_equity']:.2f}** gain=**{obj['day_gain_pct']:.1f}%** "
        f"(objetivo +{obj['daily_target_pct']:.0f}% → {obj['progress_vs_target']:.1%} del target)",
        f"- Objetivo: {obj['blurb']}",
        f"- Exits hoy: take_profit=**{tp_n}** | stop_loss=**{sl_n}** | trend_exit=**{te_n}** | time_stop=**{ts_n}**",
        f"- Learning phase: **{progress.get('suggested_phase')}**",
        f"- Exploration ratio: **{float(progress.get('exploration_ratio') or 0):.1%}**",
        f"- Current session (al generar): **{progress.get('current_session', '—')}**",
        f"- Lifetime closed trades: **{len(all_closed)}**",
        "",
        "> Nota: los patrones de aprendizaje son keys contextuales "
        "(`session|regime`, `symbol`, `exit_reason`, etc.), **no** el paquete completo "
        "de los 5 indicadores de entrada. Una loss confirmada penaliza/rechaza "
        "solo esa key, no marca BB+RSI+MACD+rejection como inválidos en bloque. "
        "Win/loss de estrategia usa **gross/TP**, no net cash; "
        "TP con PnL neto negativo = cost erosion, no loss de estrategia.",
        "",
        "## 2. Errores identificados (loss patterns)",
        f"- Confirmation threshold: **≥{threshold}** occurrences",
        "",
        "### Confirmados (ACEPTADOS)",
    ]
    lines += _pattern_decision_block(
        confirmed_losses, threshold=threshold, direction="loss", confirmed=True
    )

    lines += ["", "### Candidatos NO aceptados aún (observación)"]
    lines += _pattern_decision_block(
        observing_losses, threshold=threshold, direction="loss", confirmed=False
    )

    lines += [
        "",
        "## 3. Oportunidades (win patterns)",
        "",
        "### Confirmados (ACEPTADOS) — efecto: solo confidence boost",
    ]
    lines += _pattern_decision_block(
        confirmed_wins, threshold=threshold, direction="win", confirmed=True
    )

    lines += ["", "### Candidatos NO aceptados aún (observación)"]
    lines += _pattern_decision_block(
        observing_wins, threshold=threshold, direction="win", confirmed=False
    )

    cost_patterns = db.get_patterns(direction="cost_erosion")
    lines += [
        "",
        "## 3b. Cost erosion (TP/gross OK pero net ≤ 0 por fees/slip)",
        f"- Threshold: ≥{threshold}. Estos **no** penalizan la estrategia.",
        "",
    ]
    confirmed_cost = [p for p in cost_patterns if p["status"] == "confirmed"]
    observing_cost = [p for p in cost_patterns if p["status"] == "observing"]
    lines += ["### Confirmados"]
    lines += _pattern_decision_block(
        confirmed_cost, threshold=threshold, direction="cost_erosion", confirmed=True
    )
    lines += ["", "### En observación"]
    lines += _pattern_decision_block(
        observing_cost, threshold=threshold, direction="cost_erosion", confirmed=False
    )

    lines += [
        "",
        "## 4. Cambios implementados hoy",
        "",
        "Cada cambio lista cuántas veces se repitió el patrón antes de aplicarlo.",
    ]
    if changes:
        for c in changes:
            sess = pattern_session_name(str(c.get("pattern_key") or "")) or "—"
            lines.append(
                f"- [{c['action']}] session=`{sess}` `{c['pattern_key']}` "
                f"({c['direction']}): {c['detail']}"
            )
    else:
        lines.append("- Ninguno (sin patrones nuevos confirmados hoy).")

    lines += [
        "",
        "## 5. Progreso del aprendizaje",
        f"- Phase: **{progress.get('suggested_phase')}**",
        f"- Confirmed win patterns: **{len(confirmed_wins)}**",
        f"- Confirmed loss patterns: **{len(confirmed_losses)}**",
        f"- Capital resets (lifetime): **{resets}** | Current cash: **{cash}**",
        "",
        "### Strategy rankings",
    ]
    if rankings:
        for r in rankings:
            lines.append(
                f"- **{r['strategy']}**: exp={r['expectancy']:.4f} pf={r['profit_factor']:.2f} "
                f"n={r['trades']} status={r['status']}"
            )
    else:
        lines.append("- Sin rankings aún.")

    day_obs = [
        i
        for i in day_insights
        if i["category"]
        in ("confirmed_pattern", "capital_reset", "ml_train", "retrain", "cost_erosion")
    ]
    lines += ["", "### Insights clave del día"]
    if day_obs:
        for i in day_obs[:25]:
            lines.append(f"- [{i['category']}] {i['content']}")
    else:
        lines.append("- Sin insights clave.")

    lines += ["", "### Señales de trade rechazadas (sample)"]
    for r in rejected[:10]:
        lines.append(
            f"- {r['symbol']} reason={r['reason']} conf={r.get('confidence')}"
        )
    if not rejected:
        lines.append("- Ninguna.")

    lines += _session_report_section(
        buckets=buckets,
        day_trades=day_trades,
        cfg=cfg,
        win_patterns=win_patterns,
        loss_patterns=loss_patterns,
        changes=changes,
        threshold=threshold,
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "day": day,
        "trades": len(day_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "day_pnl": day_pnl,
        "confirmed_wins": len(confirmed_wins),
        "confirmed_losses": len(confirmed_losses),
        "changes": len(changes),
        "capital_resets": resets,
        "phase": progress.get("suggested_phase"),
        "exits": {
            "take_profit": tp_n,
            "stop_loss": sl_n,
            "trend_exit": te_n,
            "time_stop": ts_n,
        },
        "equity_day_start": obj["day_start_equity"],
        "equity_day_end": obj["current_equity"],
        "day_gain_pct": obj["day_gain_pct"],
        "daily_target_pct": obj["daily_target_pct"],
        "by_session": _session_trade_stats(day_trades, cfg),
    }
    db.save_daily_report(day, str(path), summary)
    return path


def maybe_rollover_daily_report(
    db: Database,
    learning: LearningEngine,
    last_report_day: str | None,
) -> tuple[str | None, Path | None]:
    """
    On UTC day change, write report for the previous day (complete day).
    Returns (new_last_day, path) if generated.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stored = db.get_state("last_daily_report_day")
    effective_last = last_report_day or stored

    if effective_last is None:
        db.set_state("last_daily_report_day", today)
        return today, None

    if effective_last == today:
        return None, None

    path = write_daily_report(db, learning=learning, day=effective_last)
    db.set_state("last_daily_report_day", today)
    db.insert_insight(
        "daily_report",
        f"Daily report generated for {effective_last}: {path}",
        {"day": effective_last, "path": str(path)},
    )
    return today, path
