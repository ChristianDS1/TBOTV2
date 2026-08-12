# Risk Management

## Paper soft mode (default)

- Trade **margin** ≈ €2–3 on €100; +€1 per +€50 equity (and reverse).
- **Leverage** (default **5x**): notional = margin × leverage. Fees and PnL scale on **notional** (perp-style), so paper learns real fee/PnL geometry before live.
- Max simultaneous positions: 5 (configurable).
- Correlation groups: crypto majors / USD FX pairs.
- **No** daily loss kill — bot keeps collecting data.
- Exits: take-profit (BB mid), time stop, soft liquidation, FX session end.

## Paper leverage model

| Concept | Meaning |
|---|---|
| `qty` / margin | Cash collateral locked (`base_trade_size`) |
| `leverage` | Default 5.0 |
| `notional` | margin × leverage |
| Fees | `fee_bps + slippage_bps` on **notional** each side |
| Soft liquidation | Unrealized loss ≥ `liquidation_margin_fraction` × margin → close as `liquidation` |

Fee **rate** (bps) does **not** change with leverage; higher leverage amplifies PnL and absolute fees because notional is larger.

## Auto capital refill

If paper cash/equity is exhausted and there are **no open positions**, capital is topped up to `capital_policy.refill_to` (default €100). Trading continues. Each refill is logged (`capital_reset`) and counted for the daily report.

## Fee-aware edge (forward-looking only)

Does **not** wipe learned patterns / history.

- Hard-reject only if distance to TP &lt; `hard_min_edge_multiple ×` round-trip cost
- Soft zone below `soft_min_edge_multiple`: still enters, −8 confidence (`thin_edge`)
- Take-profit close deferred unless estimated **net PnL &gt; 0** (strict; not ≥ 0). Hold until better or `time_stop` (default **10m**)

## Kill switch (technical)

Trips on:

- API failure (if enabled)
- Stale market data beyond `stale_data_seconds`
- Manual `/api/kill`

Blocks **new** entries; open positions still managed by executor. Capital refill does **not** trip the kill switch.

## Confirmed loss soft-reject

After ≥20 identical loss evidences, matching new signals can be soft-rejected (see Learning Engine). Not a capital halt.

## Live

Blocked in code until explicit future activation and statistical gates.
