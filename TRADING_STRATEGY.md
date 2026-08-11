# Trading Strategy

Primary v1 strategy: **bb_mean_reversion** from `Estrategia.txt`.

## Setup

Mean reversion after expansion/overextension inside Bollinger context.

### CALL (long)

1. Price touches / breaks lower band  
2. RSI(10) < 30  
3. MACD(5,8,9) bullish turn  
4. MACD(13,21,9) bearish but weakening  
5. Lower-wick rejection candle  

Enter on bar close when ≥ `min_conditions` (default 3). Target: BB midline. Max hold: 5 minutes.

### PUT (short)

Mirror on upper band / RSI > 70 / bearish MACD turn / upper wick.

## Soft filters (discovery)

Mid-band RSI 40–60, mid-band price, very wide bands reduce **confidence** instead of hard-blocking (see app rules on exploration).

## Horizons

`expected_holding_minutes` is per strategy. Architecture allows future intraday/swing strategies without core rewrite.
