# BULKOWSKI ENCYCLOPEDIA OF CHART PATTERNS - SPECIALIZED PROMPT FOR CURSOR / AI CODING
## Target: Short-term Trading Bot (1-5 minute candles)
## Version: 1.0 | Complete coverage of all 75 patterns from 3rd Edition + adaptation notes
## Author of original work: Thomas N. Bulkowski
## This document is a coding specification (identification rules only). It does NOT contain copyrighted statistics, performance tables or full text from the book.

---

### SYSTEM INSTRUCTIONS FOR THE AI CODER (CURSOR)

You are an expert quantitative developer building a high-precision chart pattern recognition engine for a short-term trading bot that operates exclusively on 1-minute and 5-minute candles.

Your tasks:
1. Implement robust, deterministic detectors for every pattern listed below.
2. Use pivot-based logic (local highs and local lows with configurable lookback and prominence).
3. All "near equal" price comparisons must use a relative tolerance (default 0.15% - 0.40% depending on instrument volatility; make it a parameter).
4. All time separations (bars between points) must be scaled to the timeframe (example values given for 5-min; multiply by 5 for 1-min).
5. Every pattern MUST have a confirmation rule (usually a close beyond a specific level).
6. Prefer patterns that can form in < 60-120 bars on 5-min (most useful for short-term).
7. Output for each detected pattern: pattern_name, direction (bullish/bearish), breakout_price, entry_price_suggestion, stop_price_suggestion, target_price_suggestion (measure rule), confidence_score (0-100 based on how clean the geometry is), bar_index_of_confirmation.
8. Handle both upward and downward breakouts where applicable.
9. Implement "busted pattern" logic where relevant (price breaks one way then fails and breaks the opposite way).
10. Volume is optional but if available: higher volume on breakout is a positive filter.

Common helper functions you must implement:
- find_pivots(highs, lows, left, right) → list of (index, price, type)
- is_near(price1, price2, tol_pct)
- measure_height(pattern_points)
- confirm_breakout(close, level, direction)

---

### GLOBAL PARAMETERS (make them configurable)

```python
PIVOT_LEFT = 3          # bars left for pivot
PIVOT_RIGHT = 3         # bars right for pivot
PRICE_TOL_PCT = 0.25    # % tolerance for "same price"
MIN_PATTERN_BARS = 5
MAX_PATTERN_BARS_5MIN = 120   # hard limit for short-term relevance
BREAKOUT_BUFFER_PCT = 0.05    # small buffer above/below level
```

---

## COMPLETE LIST OF PATTERNS (75 from 3rd Edition)

### 1. Harmonic Patterns (Fibonacci based - public ratios)

**AB=CD Bearish**
- Points: A (high) → B (low) → C (high) → D (low)
- AB ≈ CD in price and time (ratio 0.9-1.1)
- BC retracement of AB usually 0.382-0.886
- Confirmation: close below D
- Entry: short on close below D
- Stop: above C or above the highest of C/D
- Target: measure AB projected from C, or 1.0 / 1.272 / 1.618 extensions

**AB=CD Bullish** (mirror)
- Same ratios inverted
- Confirmation: close above D

**Bat Bearish**
- XA → AB (0.382-0.5 XA) → BC (0.382-0.886 AB) → CD (1.618-2.618 BC) ending at 0.886 XA
- Point D at 0.886 retracement of XA
- Confirmation: close below D

**Bat Bullish** (mirror)

**Butterfly Bearish**
- XA → AB (0.786 XA) → BC (0.382-0.886 AB) → CD (1.618-2.24 BC) ending at 1.27 or 1.618 XA
- Confirmation: close below D

**Butterfly Bullish** (mirror)

**Crab Bearish**
- XA → AB (0.382-0.618 XA) → BC (0.382-0.886 AB) → CD (2.24-3.618 BC) ending at 1.618 XA
- Confirmation: close below D

**Crab Bullish** (mirror)

**Gartley Bearish**
- XA → AB (0.618 XA) → BC (0.382-0.886 AB) → CD (1.13-1.618 BC) ending at 0.786 XA
- Confirmation: close below D

**Gartley Bullish** (mirror)

**Wolfe Wave Bearish**
- 5-point wave: 1-2-3-4-5 where 1-3-5 form a rising channel and 2-4 a falling channel, or classic Wolfe rules
- Point 5 is the entry zone
- Confirmation: close below the 1-3-5 trendline or below point 5
- Target: the EPA (Estimated Price at Arrival) line from 1 to 4 projected

**Wolfe Wave Bullish** (mirror)

---

### 2. Classic Reversal & Continuation Patterns

**Big M** (bearish reversal)
- Looks like a large letter M: two peaks with a lower high or similar, deep valley in middle
- Left peak ≈ right peak or right slightly lower
- Confirmation: close below the valley between the two peaks
- Best on clear prior uptrend

**Big W** (bullish reversal)
- Mirror of Big M
- Confirmation: close above the peak between the two valleys

**Broadening Bottoms**
- Expanding swings: higher highs and lower lows forming a megaphone pointing up from a bottom area
- At least 5 touches (3 on one side, 2 on the other)
- Confirmation: close above the upper trendline or below the lower (direction of breakout)

**Broadening Formation, Right-Angled and Ascending**
- Horizontal lower line + rising upper line (expanding)
- Confirmation: close beyond either line

**Broadening Formation, Right-Angled and Descending**
- Horizontal upper line + falling lower line
- Confirmation: close beyond either line

**Broadening Tops**
- Expanding megaphone after an uptrend
- Confirmation: breakout of either trendline

**Broadening Wedge, Ascending**
- Both lines rising but upper line rises faster (expanding wedge pointing up)
- Confirmation: close beyond the lines

**Broadening Wedge, Descending**
- Both lines falling, lower line falls faster
- Confirmation: close beyond the lines

**Bump-and-Run Reversal, Bottom**
- Lead-in phase (gentle rise) → Bump phase (sharp rise) → Run phase (decline that breaks the lead-in trendline)
- Confirmation: close below the lead-in trendline after the bump

**Bump-and-Run Reversal, Top**
- Mirror (sharp rise after gentle rise, then breakdown)

**Cloudbanks**
- Horizontal congestion / rectangle-like area of low volatility after a strong move, looking like a cloud bank
- Confirmation: breakout of the cloud boundaries

**Cup with Handle**
- Rounded bottom (U shape) followed by a smaller handle (pullback) on the right side
- Handle should stay in upper half of the cup
- Confirmation: close above the rim of the cup (right side of handle)
- Note: Full classic cups are rare on 1-5min; look for micro-cups (10-40 bars)

**Cup with Handle, Inverted**
- Mirror (rounded top + handle)

**Diamond Bottoms**
- Expanding then contracting diamond shape after a downtrend
- Confirmation: close above the upper boundary

**Diamond Tops**
- Expanding then contracting after uptrend
- Confirmation: close below lower boundary

**Diving Board**
- Flat base (board) followed by a sharp drop (dive) then recovery
- Confirmation: close above the board level after the dive

**Double Bottoms (4 variants)**
- Two valleys separated by a peak
- Adam = sharp V / spike
- Eve = rounded / wider
- Variants: Adam&Adam, Adam&Eve, Eve&Adam, Eve&Eve
- Price of the two bottoms within tolerance
- Rise between bottoms ≥ certain % (scaled)
- Confirmation: close above the peak between the two bottoms
- Entry: long on confirmation
- Stop: below the lower of the two bottoms
- Target: height of pattern projected upward

**Double Tops (4 variants)**
- Mirror of double bottoms (Adam/Eve combinations)
- Confirmation: close below the valley between the two peaks

**Flags**
- Strong sharp move (flagpole) followed by a small rectangular / parallel channel consolidation sloping against the prior move
- Duration short (5-20 bars on 5min ideal)
- Confirmation: close beyond the flag in the direction of the pole
- Target: length of the flagpole projected from breakout

**Flags, High and Tight**
- Extremely strong vertical rise (almost no pullback) followed by a tight flag
- Very powerful continuation
- Confirmation: breakout of the tight consolidation

**Gaps**
- Types: common, breakaway, continuation (runaway), exhaustion
- Detection: open of current bar significantly away from previous close with no overlap
- Breakaway after consolidation + volume → continuation signal
- Exhaustion after long run → possible reversal

**Head-and-Shoulders Bottoms**
- Three valleys: left shoulder, lower head, right shoulder (shoulders roughly equal height and distance)
- Neckline connecting the two peaks between shoulders and head
- Confirmation: close above the neckline
- Complex version: multiple shoulders or heads

**Head-and-Shoulders Tops**
- Mirror (left shoulder, higher head, right shoulder)
- Confirmation: close below neckline
- Complex version exists

**Horn Bottoms**
- Two downward spikes (horns) separated by a short rise, looking like a U with spikes
- Confirmation: close above the high between the horns

**Horn Tops**
- Mirror

**Island Reversals**
- Price gaps away from a congestion, then gaps back, leaving an island of bars isolated by gaps on both sides
- Confirmation: the second gap in the opposite direction

**Measured Move Down**
- Sharp decline (first leg) → corrective bounce → second decline of similar length
- Target of second leg ≈ length of first leg

**Measured Move Up**
- Mirror

**Pennants**
- Similar to flags but the consolidation is a small symmetrical triangle (converging)
- Short duration
- Confirmation + target same as flags (flagpole projection)

**Pipe Bottoms**
- Two adjacent downward spikes (or very close) of similar length, looking like a pipe
- Confirmation: close above the high of the pipe

**Pipe Tops**
- Mirror

**Rectangle Bottoms**
- Horizontal support and resistance (flat range) after a downtrend
- At least 2 touches each side
- Confirmation: close above resistance (bullish) or below support (bearish)

**Rectangle Tops**
- Same geometry after uptrend

**Roof**
- Horizontal top + rising bottom (like a roof)
- Confirmation: breakout

**Roof, Inverted**
- Horizontal bottom + falling top

**Rounding Bottoms**
- Gradual U-shaped turn (saucer)
- Confirmation: close above the right rim

**Rounding Tops**
- Gradual inverted U

**Scallops, Ascending**
- Series of rising rounded bottoms (like a series of J or scallops)
- Confirmation: breakout of the pattern high

**Scallops, Ascending and Inverted**
- Rising inverted scallops

**Scallops, Descending**
- Falling rounded tops

**Scallops, Descending and Inverted**
- Falling inverted scallops

**Three Falling Peaks**
- Three successive lower peaks
- Confirmation: close below the lowest valley between them

**Three Peaks and Domed House**
- Three peaks followed by a rounded "domed house" structure
- Complex reversal

**Three Rising Valleys**
- Three successive higher valleys
- Confirmation: close above the highest peak between them

**Triangles, Ascending**
- Flat resistance + rising support
- Confirmation: close above resistance (most common) or below support

**Triangles, Descending**
- Flat support + falling resistance
- Confirmation: close below support (most common)

**Triangles, Symmetrical**
- Converging higher lows and lower highs
- Confirmation: close beyond either trendline
- Direction often continuation of prior trend

**Triple Bottoms**
- Three valleys at approximately the same price
- Confirmation: close above the highest peak between the valleys

**Triple Tops**
- Three peaks at approximately the same price
- Confirmation: close below the lowest valley between the peaks

**V-Bottoms**
- Sharp V reversal (almost no rounding)
- Confirmation: close above the high of the V or a short-term swing high after the low

**V-Bottoms, Extended**
- V bottom followed by a consolidation or extension before continuation

**V-Tops**
- Sharp inverted V
- Confirmation: close below the low of the V

**V-Tops, Extended**
- Mirror of extended V-bottom

**Wedges, Falling**
- Both trendlines falling, converging (bullish usually)
- Confirmation: close above upper line

**Wedges, Rising**
- Both trendlines rising, converging (bearish usually)
- Confirmation: close below lower line

---

### IMPLEMENTATION PRIORITY FOR 1-5 MIN BOT (recommended order)

**High priority (frequent + useful on short TF):**
Flags, Pennants, Rectangles, Ascending/Descending/Symmetrical Triangles, Double Bottoms/Tops (all 4), Triple Bottoms/Tops, Head & Shoulders (simple), Gaps, V-Bottoms/Tops, Pipe Bottoms/Tops, Horns, Three Rising Valleys / Three Falling Peaks, Falling/Rising Wedges

**Medium priority:**
Broadening formations (short versions), Scallops (micro), Diamond, Island Reversals, Bump-and-Run (if clear), Measured Moves (short)

**Low priority / rare on 1-5min:**
Full Cup with Handle, large Rounding, long harmonics (Bat, Crab, etc. only if very clean micro versions), Cloudbanks, Roof, Diving Board, Three Peaks and Domed House, Big M / Big W (only micro)

---

### CODING QUALITY RULES

- Never force a pattern. If geometry is ambiguous → do not detect.
- Prefer fewer false positives over high recall.
- Always calculate a "geometry quality score" (how well the points fit the ideal ratios/angles).
- Log every detection with the exact pivot indices used.
- Support both live streaming detection and historical backtest mode.
- Make every tolerance and bar-count parameter configurable.

---

### FINAL CHECKLIST FOR YOU (the AI coder)

After implementing, verify:
1. All 75 patterns from the list above have at least a basic detector skeleton.
2. Confirmation logic is present for every pattern.
3. Entry / Stop / Target suggestions are generated.
4. Timeframe scaling is applied.
5. Pivot detection is robust.
6. No hardcoded magic numbers without parameters.
7. Code is clean, typed, and documented.

This specification is complete and covers every pattern in the 3rd edition of Encyclopedia of Chart Patterns by Thomas N. Bulkowski, adapted for short-term (1-5 min) algorithmic trading.

END OF PROMPT
