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

Target: BB midline. Max hold: **10** minutes (1–10m scalping window).

### PUT (short)

Mirror on upper band / RSI > 70 / bearish MACD turn / upper wick.

## Soft filters (discovery)

Mid-band RSI 40–60, mid-band price, very wide bands reduce **confidence** instead of hard-blocking (see app rules on exploration). Being too far from the extreme is a **hard skip**, not a soft penalty.

## Horizons

`expected_holding_minutes` / `max_hold_minutes` per strategy. Architecture allows future intraday/swing strategies without core rewrite.
