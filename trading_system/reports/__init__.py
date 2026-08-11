"""Daily learning report helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.config import ROOT
from trading_system.database import Database
from trading_system.learning import LearningEngine


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

    wins = [t for t in day_trades if (t.pnl or 0) > 0]
    losses = [t for t in day_trades if (t.pnl or 0) <= 0]
    day_pnl = sum(t.pnl or 0 for t in day_trades)
    wr = len(wins) / len(day_trades) if day_trades else 0.0
    exp = day_pnl / len(day_trades) if day_trades else 0.0

    confirmed_wins = [p for p in win_patterns if p["status"] == "confirmed"]
    confirmed_losses = [p for p in loss_patterns if p["status"] == "confirmed"]
    observing_wins = [p for p in win_patterns if p["status"] == "observing"]
    observing_losses = [p for p in loss_patterns if p["status"] == "observing"]

    threshold = 20
    if learning is not None:
        threshold = learning.cfg.pattern_min_occurrences
        progress = learning.phase_progress(len(all_closed))
    else:
        progress = {
            "suggested_phase": "unknown",
            "exploration_ratio": 0,
            "confirmed_wins": len(confirmed_wins),
            "confirmed_losses": len(confirmed_losses),
        }

    resets = int(db.get_state("capital_resets") or "0")
    cash = db.get_state("cash") or "?"

    lines = [
        f"# Daily Learning Report — {day}",
        "",
        "## 1. Aprendizaje del día",
        f"- Closed trades today: **{len(day_trades)}** (W {len(wins)} / L {len(losses)})",
        f"- Win rate today: **{wr:.1%}** | Expectancy: **{exp:.4f}** | Day PnL: **{day_pnl:.4f}**",
        f"- Learning phase: **{progress.get('suggested_phase')}**",
        f"- Exploration ratio: **{float(progress.get('exploration_ratio') or 0):.1%}**",
        f"- Lifetime closed trades: **{len(all_closed)}**",
        "",
        "> Nota: los patrones de aprendizaje son keys contextuales "
        "(`regime`, `symbol`, `exit_reason`, etc.), **no** el paquete completo "
        "de los 5 indicadores de entrada. Una loss confirmada penaliza/rechaza "
        "solo esa key, no marca BB+RSI+MACD+rejection como inválidos en bloque.",
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

    lines += [
        "",
        "## 4. Cambios implementados hoy",
        "",
        "Cada cambio lista cuántas veces se repitió el patrón antes de aplicarlo.",
    ]
    if changes:
        for c in changes:
            lines.append(
                f"- [{c['action']}] `{c['pattern_key']}` ({c['direction']}): {c['detail']}"
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
        in ("confirmed_pattern", "capital_reset", "ml_train", "retrain")
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
