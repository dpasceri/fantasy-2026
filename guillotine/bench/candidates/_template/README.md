# _template (copy me)

1. Copy this folder to `bench/candidates/<key>/`. Use a valid Python identifier for the key,
   because the folder is imported as `bench.candidates.<key>`.
2. In both run scripts, set `KEY=<key>` and the `--candidate module:Class` / `--variant` flags.
3. Write this README:
   - **What**: the method, in a few lines.
   - **Why**: why it might beat the baseline here.
   - **Sources**: papers and libraries, with links.
   - **Variants**: each `--variant` name and what it changes.
4. Run the smoke test through cpuq: `bench/cpuq -n 1 -- bash bench/candidates/<key>/run_smoke.sh`.
   It must take at most 90 s on 1 core and write `bench/results/<key>.smoke.json`.
5. Check it with `python -m bench.harness.compare player --mode smoke`.
6. Do not run `run_full.sh` yourself (at most 15 min on 6 cores). Enqueue it instead:

   ```
   bench/queue/pending/<key>.job = {"key": "<key>", "cmd": ["bash", "bench/candidates/<key>/run_full.sh"], "timeout_s": 1200}
   ```

`candidate.py` is a tiny player-track example: additive residual quantiles by position, with
variants `additive` and `recent`. Fit only on `train_df`, which is everything strictly before the
test week. Any tuning (inner validation, variant choice) must also use only `train_df`.

Player-track rules of thumb:

- Return rows in `test_df` order.
- Seed every RNG from `bench.harness.task_seed()`.
- Size thread pools with `bench.harness.task_threads()`.
- Never read `data/*.json` or `common.League` (`info`, `proj`) directly: they hold today's team
  and injury tags and later weeks.
- `bench.harness.data.table()` and its helpers are safe, because inside the harness they stop
  at the test week's kickoff.
