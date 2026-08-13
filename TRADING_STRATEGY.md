# Trading Strategy

Primary v1 strategy: **bb_mean_reversion** from `Estrategia.txt`.

## Setup

Mean reversion after expansion/overextension inside Bollinger context. Enter near the **extreme band** on early rejection — not after price has already traveled to the midline.

### CALL (long)

1. Price touches / breaks lower band  
2. RSI(10) < 30  
3. MACD(5,8,9) bullish turn  
4. MACD(13,21,9) bearish but weakening  
5. Lower-wick rejection candle  

Enter when ≥ `min_conditions` (default 3) **and**:

- `require_rejection_candle: true` → rejection candle is a hard gate  
- close still near the extreme (`max_extreme_retrace_pct`, default 0.35 of band width toward mid)

**Take profit / stop:**
- TP default `tp_mode: trend_fade` — **no fixed TP price**; exit as `trend_exit` when opposing rejection + momentum fade score ≥ `trend_fade_min_score` (default 2) and estimated net &gt; 0
- Fee-edge at entry still uses distance to **BB mid** as expected-move proxy (not a hard target)
- SL default `sl_mode: margin_pct` — max **NET** loss = `sl_margin_pct` of margin (default **4%** → €0.40 on margin 10); exit fees included in the budget
- Paper defaults: margin `base_trade_size` 10, `leverage` 20
- Legacy: `tp_mode: band_fraction|fixed_bps`, `sl_mode: rr_from_tp|band`

**Hold:** no `time_stop`. Positions stay open until `trend_exit` (fade, net &gt; 0), `stop_loss` (4% margin NET), liquidation, or FX `session_end`. `min_hold_minutes` still gates fade. `max_hold_minutes` is metadata only.

### PUT (short)

Mirror on upper band / RSI > 70 / bearish MACD turn / upper wick.

## Soft filters (discovery)

Mid-band RSI 40–60, mid-band price, very wide bands reduce **confidence** instead of hard-blocking (see app rules on exploration). Being too far from the extreme is a **hard skip**, not a soft penalty.

## Horizons

`expected_holding_minutes` / `max_hold_minutes` per strategy. Architecture allows future intraday/swing strategies without core rewrite.
