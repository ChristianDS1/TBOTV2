# Risk Management

## Paper soft mode (default)

- Trade size ≈ €2–3 on €100; +€1 per +€50 equity (and reverse).
- Max simultaneous positions: 5 (configurable).
- Correlation groups: crypto majors / USD FX pairs.
- **No** daily loss kill — bot keeps collecting data.
- Exits: take-profit (BB mid), time stop, FX session end.

## Auto capital refill

If paper cash/equity is exhausted and there are **no open positions**, capital is topped up to `capital_policy.refill_to` (default €100). Trading continues. Each refill is logged (`capital_reset`) and counted for the daily report.

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
