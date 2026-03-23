# Profitability Overhaul Plan — 7 Fixes to Positive Expectancy

**Created**: 2026-03-20
**Goal**: Transform system from negative R:R (0.4:1) to positive expectancy (1.0-1.5:1)
**Account**: Apex 50K ($2,500 trailing drawdown, 50 MNQ scaling → 100 after +$2,600)
**Method**: Implement each fix → audit 100/100 → next fix

---

## Step 1: Partial Profit Taking (HIGHEST IMPACT)

**Why**: Single biggest R:R fix. Currently all-or-nothing 2 contracts. Partials lock in profit on contract 1, let contract 2 run.

### Changes Required

**File: `src/execution/tick_stop_monitor.py`**
- Add `_partial_taken: bool` tracking field
- Add `_partial_quantity: int` field (default 1 = half of 2-contract entry)
- Add `_partial_target_points: float` field (configurable, default 15.0 pts)
- New method `_check_partial_profit(price) -> bool`
  - For long: `price >= entry_price + partial_target_points`
  - For short: `price <= entry_price - partial_target_points`
- In `on_trade()`, between trail update and stop check:
  ```python
  if not self._partial_taken and self._check_partial_profit(price):
      self._partial_taken = True
      await self._execute_partial(price)
  ```
- New method `_execute_partial(price)`:
  - Calls a NEW callback `partial_fn` (not flatten_fn) to close 1 contract
  - Moves stop to breakeven + 1 pt on remaining contract
  - Logs: `tick_stop_monitor.PARTIAL_TAKEN`
- Modify `activate()`: accept `partial_target_points`, `partial_quantity`, `partial_fn` params
- After partial: trail continues on remaining 1 contract as before

**File: `src/execution/quantlynk_client.py`**
- Add `partial_close(quantity: int, price: float)` method
  - For long entry: sends `sell` with quantity=1
  - For short entry: sends `buy` with quantity=1
- This is NOT a flatten — it's a directional close of 1 contract

**File: `src/orchestrator.py`**
- In `_execute_confluence_action()` where tick_stop_monitor is activated (~line 959):
  - Pass `partial_fn=quantlynk.partial_close` (or wrapped lambda)
  - Pass `partial_target_points` from config
- In post-exit bookkeeping: handle partial PnL tracking

**File: `src/core/config.py`**
- Add to TradingConfig:
  ```python
  partial_profit_points: float = 15.0  # take 1st contract off at +15pts
  partial_quantity: int = 1            # close 1 of 2 contracts
  partial_breakeven_offset: float = 1.0  # move stop to entry + 1pt after partial
  ```
- Add ETH variants:
  ```python
  eth_partial_profit_points: float = 10.0  # tighter for ETH ranges
  ```

**File: `src/core/config.py` (SessionRiskParams)**
- Add `partial_target` per session to override default:
  - Asian: 10 pts (smaller ranges)
  - London: 12 pts
  - RTH: 15 pts

### Tests Required
- `test_partial_taken_at_target`: long entry at 100, partial fires at 115, verify callback
- `test_partial_not_taken_below_target`: price reaches 114.75, no partial
- `test_stop_moves_to_breakeven_after_partial`: verify stop = entry + 1
- `test_trail_continues_after_partial`: runner still trails normally
- `test_partial_short_direction`: short entry partial calls buy(1)
- `test_no_double_partial`: partial flag prevents re-fire
- `test_eth_partial_target`: uses eth_partial_profit_points
- `test_partial_persists_to_disk`: state survives restart

### Audit Criteria (100/100)
- [ ] (20) Partial callback fires at exact target — verified with mock
- [ ] (15) Stop moves to breakeven+1 after partial — verified both long/short
- [ ] (15) Trail continues on runner — no regression in 3-tier trail
- [ ] (10) QuantLynk sends correct directional close (sell for long partial, buy for short)
- [ ] (10) State persisted to disk and restored on restart
- [ ] (10) Config params respected: session-specific targets, ETH override
- [ ] (10) Logging complete: partial_taken, new_stop, remaining_quantity
- [ ] (5) No existing tests broken
- [ ] (5) Edge cases: price gaps through target, partial during grace period

---

## Step 2: Block/Reduce Size in FAST Markets

**Why**: FAST speed state = speed_ratio > 1.5x baseline. This is news spikes, flash crashes — worst time for mechanical entries. Currently only a risk_flag (warning), not a block.

### Changes Required

**File: `src/indicators/confluence.py`**
- In `score()` method, where FAST is handled (~line 236):
  ```python
  # CURRENT (broken):
  elif speed_state == "FAST":
      risk_flags.append("fast_market")

  # NEW:
  elif speed_state == "FAST":
      risk_flags.append("fast_market")
      # Don't hard-block — reduce score by 1 and flag for size reduction
      fast_market_penalty = True
  ```
- After total_score computed: `if fast_market_penalty: total_score = max(0, total_score - 1)`
- Add `fast_market` flag to return dict

**File: `src/orchestrator.py`**
- In `_confluence_decision_cycle()`, after confluence passes:
  ```python
  if best_result.get("fast_market"):
      # Reduce to 1 contract in fast markets (half normal size)
      entry_quantity = 1
      logger.info("confluence.fast_market_size_reduction")
  ```

**File: `src/execution/risk_manager.py`**
- Add `FAST_MARKET` as a recognized risk flag in entry check logging
- No hard block — the score penalty + size reduction is sufficient

### Tests Required
- `test_fast_market_score_penalty`: FAST reduces score by 1
- `test_fast_market_reduces_quantity`: entry uses 1 contract not 2
- `test_fast_market_flag_in_result`: result dict contains fast_market=True
- `test_normal_speed_no_penalty`: NORMAL speed has no score reduction
- `test_fast_market_still_enters_high_confluence`: score 5 - 1 = 4, still passes min 3

### Audit Criteria (100/100)
- [ ] (25) FAST market reduces score by 1 — verified in unit test
- [ ] (25) FAST market reduces entry size to 1 contract — verified in orchestrator
- [ ] (20) No hard block on FAST (legitimate setups still trade, just smaller)
- [ ] (15) NORMAL and SLOW states unchanged — no regression
- [ ] (10) Logging captures fast market flag and size reduction
- [ ] (5) Pre-market LLM no_trade_windows still work for scheduled events

---

## Step 3: Dynamic Stop Loss (OB Zone Edge + ATR Buffer)

**Why**: Static 40pt stop is 3x wider than necessary. ICT entries have a natural invalidation point — the other edge of the OB zone. Dynamic stops cut average loss by 40-60%.

### Changes Required

**File: `src/indicators/confluence.py`**
- In `_score_order_block()`: when OB tap is scored (2 pts), also return the OB zone boundaries:
  ```python
  return {
      "score": 2,
      "detail": f"...",
      "ob_zone_low": ob.low,
      "ob_zone_high": ob.high,
      "ob_side": ob.side,
  }
  ```
- In `score()` return dict: include `ob_zone` if present

**File: `src/execution/risk_manager.py`**
- New method `compute_dynamic_stop(side, entry_price, ob_zone, atr, session_params) -> float`:
  ```python
  def compute_dynamic_stop(self, side, entry_price, ob_zone, atr, phase):
      session_params = self.get_params(phase)
      max_sl = session_params.sl_points  # 40pt hard cap (safety net)

      if ob_zone:
          # Stop = other side of OB + ATR buffer
          if side == "long":
              dynamic_sl = entry_price - ob_zone["ob_zone_low"] + (atr * 0.5)
          else:
              dynamic_sl = ob_zone["ob_zone_high"] - entry_price + (atr * 0.5)

          # Clamp: minimum 8pts (avoid getting stopped on noise), max session limit
          dynamic_sl = max(8.0, min(dynamic_sl, max_sl))
      else:
          # No OB zone (entered on sweep/candle only) — use ATR-based stop
          dynamic_sl = max(10.0, min(atr * 3.0, max_sl))

      return round(dynamic_sl, 2)
  ```

**File: `src/orchestrator.py`**
- In `_confluence_decision_cycle()`, replace static `sl_pts = self._risk_manager.get_sl_points(phase)`:
  ```python
  ob_zone = best_result.get("ob_zone")
  sl_pts = self._risk_manager.compute_dynamic_stop(
      side=best_side,
      entry_price=state.last_price,
      ob_zone=ob_zone,
      atr=state.atr,
      phase=phase,
  )
  ```
- Log the dynamic stop: `dynamic_sl_pts=sl_pts, ob_zone=ob_zone, atr=state.atr`

**File: `src/core/config.py`**
- Add to TradingConfig:
  ```python
  min_stop_points: float = 8.0    # never tighter than 8pts
  atr_stop_multiplier: float = 3.0  # fallback: 3x ATR when no OB
  ob_stop_buffer_atr: float = 0.5   # buffer beyond OB edge
  ```

### Tests Required
- `test_dynamic_stop_long_ob_zone`: OB at 100-105, entry at 104, stop = 100 - 0.5*ATR
- `test_dynamic_stop_short_ob_zone`: OB at 110-115, entry at 111, stop = 115 + 0.5*ATR
- `test_dynamic_stop_clamped_min`: narrow OB produces stop of 5pts → clamped to 8
- `test_dynamic_stop_clamped_max`: wide OB produces stop of 50pts → clamped to 40
- `test_dynamic_stop_no_ob`: entry without OB uses ATR*3 fallback
- `test_dynamic_stop_atr_fallback_clamped`: ATR fallback respects min/max
- `test_widen_stops_applies_to_dynamic`: pre-market widen_stops multiplies dynamic stop too

### Audit Criteria (100/100)
- [ ] (20) Dynamic stop uses OB zone edge + ATR buffer — verified long and short
- [ ] (15) Min clamp 8pts prevents noise stop-outs — verified
- [ ] (15) Max clamp preserves session hard limit as safety net — verified
- [ ] (15) Fallback to ATR*3 when no OB zone — verified
- [ ] (10) OB zone data flows from confluence → orchestrator → risk_manager → tick_stop
- [ ] (10) widen_stops applies multiplicatively to dynamic stop too
- [ ] (10) Logging shows dynamic_sl, ob_zone, atr at entry
- [ ] (5) No regression in existing stop tests

---

## Step 4: Direction-Aware Volume Scoring

**Why**: High sell volume currently gives +1 confluence to a long signal. Volume should confirm the trade direction.

### Changes Required

**File: `src/indicators/confluence.py`**
- Modify `_score_volume()` to accept `side` parameter
- Access buy_volume/sell_volume from bars (already computed in state_engine):
  ```python
  @staticmethod
  def _score_volume(side: str, bars_1m: list[dict], atr: float) -> dict:
      if len(bars_1m) < VOLUME_SMA_PERIOD:
          return {"score": 0, "detail": "insufficient bars"}

      recent = bars_1m[-VOLUME_SMA_PERIOD:]
      volumes = [b.get("volume", 0) for b in recent]
      sma = sum(volumes) / len(volumes)
      current_vol = volumes[-1]

      # Must have above-average volume
      if current_vol < sma * VOLUME_SPIKE_MULT:
          return {"score": 0, "detail": f"volume {current_vol} < {sma * VOLUME_SPIKE_MULT:.0f} threshold"}

      # Direction check: buy_volume vs sell_volume
      current_bar = bars_1m[-1]
      buy_vol = current_bar.get("buy_volume", 0)
      sell_vol = current_bar.get("sell_volume", 0)
      total_dir = buy_vol + sell_vol

      if total_dir > 0:
          if side == "long" and buy_vol < sell_vol:
              return {"score": 0, "detail": f"volume spike but sell-dominated ({sell_vol}>{buy_vol})"}
          if side == "short" and sell_vol < buy_vol:
              return {"score": 0, "detail": f"volume spike but buy-dominated ({buy_vol}>{sell_vol})"}

      return {"score": 1, "detail": f"volume confirmed: {current_vol} >= {sma * VOLUME_SPIKE_MULT:.0f}, direction aligned"}
  ```
- Update `score()` caller to pass `side` to `_score_volume()`

**File: `src/data/state_engine.py`**
- Verify `buy_volume` and `sell_volume` are available in bar dicts
- If not present (e.g., from historical data), gracefully skip direction check

### Tests Required
- `test_volume_long_buy_dominated`: high volume + buy > sell = 1 pt for long
- `test_volume_long_sell_dominated`: high volume + sell > buy = 0 pts for long
- `test_volume_short_sell_dominated`: high volume + sell > buy = 1 pt for short
- `test_volume_short_buy_dominated`: high volume + buy > sell = 0 pts for short
- `test_volume_no_directional_data`: no buy/sell fields → fall back to total volume only
- `test_volume_below_sma`: low volume = 0 regardless of direction

### Audit Criteria (100/100)
- [ ] (25) Direction-aligned volume scores 1, counter-direction scores 0 — verified both sides
- [ ] (25) Graceful degradation when buy/sell volume unavailable — verified
- [ ] (20) Volume SMA threshold still required (direction alone insufficient) — verified
- [ ] (15) No regression in existing volume tests
- [ ] (10) Detail string clearly explains why volume scored or didn't
- [ ] (5) Edge case: buy_vol == sell_vol treated as "direction aligned" (neutral is OK)

---

## Step 5: Order Block Time Decay

**Why**: 4-hour-old OBs on 1-min charts are stale. Institutional flow is long gone. They should score less over time.

### Changes Required

**File: `src/indicators/confluence.py`**
- Add `created_at: float` field to `OrderBlock` dataclass (epoch timestamp)
- In `_detect_order_blocks()`: set `created_at = time.time()` on new OBs
- Modify `_score_order_block()`:
  ```python
  def _score_order_block(self, side, last_price, bars_1m, atr):
      # ... existing OB matching ...
      for ob in obs:
          if ob.low <= last_price <= ob.high:
              age_minutes = (time.time() - ob.created_at) / 60.0

              if age_minutes <= 30:
                  score = 2    # fresh OB — full score
              elif age_minutes <= 60:
                  score = 1    # aging OB — half score
              else:
                  score = 0    # stale OB — no score (mitigation target, not entry)
                  # Don't return yet — remove stale OB from active list
                  continue

              return {"score": score, "detail": f"... (age={age_minutes:.0f}min)"}
  ```
- Add periodic OB cleanup: remove OBs older than 120 minutes in `update()`

### Tests Required
- `test_ob_fresh_scores_2`: OB created 10 min ago scores 2
- `test_ob_aging_scores_1`: OB created 45 min ago scores 1
- `test_ob_stale_scores_0`: OB created 90 min ago scores 0
- `test_ob_cleanup_removes_old`: OBs older than 120 min are pruned
- `test_ob_age_tracks_correctly`: verify created_at is set on detection
- `test_ob_no_created_at_graceful`: legacy OBs without timestamp treated as fresh (safe default)

### Audit Criteria (100/100)
- [ ] (25) Fresh OB (≤30min) scores 2, aging (30-60min) scores 1, stale (>60min) scores 0
- [ ] (20) Stale OBs pruned from active list after 120 min
- [ ] (15) created_at timestamp set on detection — verified
- [ ] (15) Graceful handling of OBs without timestamp (legacy/restart)
- [ ] (10) Age shown in log detail string
- [ ] (10) No regression in existing OB tests
- [ ] (5) Disk persistence includes created_at field

---

## Step 6: Sweep "Already Used" Flag

**Why**: Same sweep level fires on every bar until price moves away. The first sweep is valid; subsequent ones are noise.

### Changes Required

**File: `src/indicators/confluence.py`**
- Add `swept: bool = False` and `swept_at: float = 0.0` fields to `SweepLevel` dataclass
- In `_score_liquidity_sweep()`:
  ```python
  # Before scoring a sweep, check if already used
  if level_obj.swept:
      continue  # Skip — already triggered on this level

  # ... existing sweep confirmation logic ...

  if swept and reversed:
      # Mark as used
      level_obj.swept = True
      level_obj.swept_at = time.time()
      return {"score": 1, "detail": f"sweep confirmed at {level:.2f} (first touch)"}
  ```
- Sweep levels reset `swept = False` if price moves away by > 1.5 * ATR (level becomes relevant again after significant move)
- Session level sweeps (asian_high, etc.) are stateless — tracked separately with a `_used_session_sweeps: set[str]` that resets on session change

### Tests Required
- `test_sweep_fires_once`: first sweep at level scores 1
- `test_sweep_does_not_refire`: same level on next bar scores 0
- `test_sweep_resets_after_distance`: price moves 2*ATR away, level reset, can fire again
- `test_session_sweep_fires_once`: asian_low sweep fires once per session
- `test_session_sweep_resets_on_new_session`: new session clears used sweeps
- `test_multiple_sweep_levels_independent`: sweeping level A doesn't mark level B

### Audit Criteria (100/100)
- [ ] (25) Sweep fires exactly once per level — verified with consecutive bars
- [ ] (20) Reset logic works: price moves 1.5*ATR away → level available again
- [ ] (20) Session sweeps tracked separately and reset on session change
- [ ] (15) No regression in existing sweep detection tests
- [ ] (10) swept/swept_at fields persist and restore correctly
- [ ] (10) Detail string shows "(first touch)" vs "(already swept)"

---

## Step 7: Dynamic Position Scaling (Apex-Aware)

**Why**: Hard-coded 2-contract entries ignore Apex's 50-100 contract allowance. As account grows, position size should grow proportionally within risk limits.

### Changes Required

**File: `src/orchestrator.py`**
- Remove the hard cap of `max_entry = 2` at line 1590
- Replace with dynamic sizing:
  ```python
  def _compute_entry_size(self, sl_points: float, phase: SessionPhase) -> int:
      """Compute position size based on risk budget and dynamic stop.

      Risk per trade = 1% of trailing drawdown remaining.
      Size = risk_budget / (sl_points * point_value)
      Clamped by: session_controller.effective_max_contracts, apex max_micros
      """
      # Risk budget: 1% of drawdown remaining (conservative)
      dd_remaining = self._apex_state.drawdown_remaining if self._apex_state else 2500.0
      risk_budget = dd_remaining * 0.01  # $25 per trade with $2,500 remaining

      # Minimum viable risk: at least $10 (1 contract with tight stop)
      if risk_budget < 10.0:
          return 1

      point_value = self._config.trading.point_value  # $2.0 for MNQ
      contracts_by_risk = int(risk_budget / (sl_points * point_value))

      # Clamp by all limits
      max_from_session = self._session_ctrl.effective_max_contracts
      max_from_apex = self._apex_state.effective_max_micros if self._apex_state else 6
      max_entry = min(contracts_by_risk, max_from_session, max_from_apex)

      # During scaling phase (before $2,600 profit): half max
      if self._apex_state and not self._apex_state.scaling_unlocked:
          max_entry = min(max_entry, self._apex_state.max_micros_scaling)

      # Floor: always at least 1
      return max(1, max_entry)
  ```

**File: `src/core/config.py`**
- Add: `risk_per_trade_pct: float = 0.01  # 1% of drawdown remaining`
- Add: `max_entry_contracts: int = 10  # hard ceiling per entry regardless of math`

**File: `src/execution/tick_stop_monitor.py`**
- Partial quantity = total_quantity // 2 (not hardcoded 1)
- If entry is 4 contracts: partial = 2, runner = 2

**File: `src/guardrails/apex_rules.py`**
- Add `effective_max_micros` property to `ApexAccountState`:
  ```python
  @property
  def effective_max_micros(self) -> int:
      if self.scaling_unlocked:
          return self.max_micros  # 100
      return self.max_micros_scaling  # 50
  ```

### Scaling Progression (Apex 50K)

| Drawdown Remaining | Risk Budget (1%) | Dynamic Stop (15pts) | Entry Size | Risk/Trade |
|---|---|---|---|---|
| $2,500 (fresh) | $25 | 15 pts | 1 MNQ | $30 |
| $2,500 (fresh) | $25 | 10 pts | 1 MNQ | $20 |
| After +$2,600 (unlocked) | $50+ | 15 pts | 1-2 MNQ | $30-60 |
| After +$5,000 | $75 | 15 pts | 2 MNQ | $60 |
| After +$10,000 | $125 | 15 pts | 4 MNQ | $120 |
| After +$20,000 | $225 | 15 pts | 7 MNQ | $210 |

**Note**: This is conservative. With dynamic stops averaging 15pts instead of 40pts, 1% risk allows more contracts because each contract risks less.

### Tests Required
- `test_scaling_fresh_account`: $2,500 DD, 15pt stop → 1 contract
- `test_scaling_after_unlock`: $5,100 DD, 15pt stop → 2 contracts
- `test_scaling_profitable_account`: $12,500 DD, 15pt stop → 4 contracts
- `test_scaling_respects_apex_max`: never exceed apex max_micros
- `test_scaling_respects_session_max`: never exceed effective_max_contracts
- `test_scaling_tight_stop_more_contracts`: 8pt stop allows more than 15pt
- `test_scaling_wide_stop_fewer_contracts`: 25pt stop reduces size
- `test_partial_scales_with_entry`: 4-contract entry → partial = 2, runner = 2
- `test_low_drawdown_remaining_min_1`: near-blown account still trades 1 contract

### Audit Criteria (100/100)
- [ ] (20) Risk-based sizing formula correct — verified with multiple scenarios
- [ ] (15) Apex max_micros respected (scaling locked vs unlocked)
- [ ] (15) Session controller effective_max respected
- [ ] (15) Partial quantity scales (half of entry, not hardcoded 1)
- [ ] (10) Hard ceiling (max_entry_contracts) prevents over-sizing
- [ ] (10) 1% risk per trade = conservative enough for $2,500 drawdown survival
- [ ] (10) Fresh account starts at 1 contract (survival mode)
- [ ] (5) Config parameters documented and overridable

---

## Execution Order & Dependencies

```
Step 1 (Partials) ──→ Step 3 (Dynamic Stops) ──→ Step 7 (Scaling)
                         ↑                           ↑
Step 2 (FAST block) ────┘                           |
Step 4 (Volume direction) ──────────────────────────┘
Step 5 (OB decay) ──────────────────────────────────┘
Step 6 (Sweep flag) ────────────────────────────────┘
```

**Recommended order**: 1 → 2 → 3 → 4 → 5 → 6 → 7

Steps 4, 5, 6 are independent signal quality fixes — order among them doesn't matter.
Step 7 depends on Step 1 (partial quantity must scale) and Step 3 (dynamic stop feeds sizing formula).

---

## Post-Implementation Validation

After all 7 steps at 100/100:
- Run full test suite (expect 1550+ tests)
- Deploy to VPS
- Monitor first 5 trading days:
  - Verify dynamic stops are 10-20pts (not 40)
  - Verify partials firing at targets
  - Verify FAST market size reduction
  - Verify volume direction filtering
  - Verify OB decay working
  - Verify sweep single-fire
  - Verify position sizing scaling with account
- Calculate realized R:R from trade journal

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Avg loss | $160 (40pts × 2 × $2) | $40-60 (12-15pts × 1-2 × $2) |
| Avg win | $80 (trail exit ~20pts × 2 × $2) | $70-130 (partial + runner) |
| R:R | 0.4:1 | 1.0-1.5:1 |
| Required win rate | 73% | 45-50% |
| False signals filtered | 0 | FAST + stale OB + counter-volume + repeat sweeps |
| Drawdown survival (losses before blown) | 15 | 40-60 |
| Scaling path | Fixed 2 MNQ forever | 1 → 10+ MNQ as account grows |
