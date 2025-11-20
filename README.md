# F1 Race Strategy Simulator

Monte Carlo simulator for Formula 1 race strategies. The tool models tire degradation, fuel load effects, and stochastic lap-time noise to compare multi-stint race plans.

## Features

- Polynomial tire degradation tuned per compound with thermal sensitivity
- Lap-time model combining base pace, fuel, tire temps, track evolution,
  opponent-based undercut, and stochastic effects
- Built-in 1-stop, 2-stop, and 3-stop strategy templates
- Weather system (sunny/cloudy/rain), safety-car odds, pit-stop timing jitter,
  plus DRS/ERS deployment modeling
- Driver skill presets (elite or average) that influence lap-time variance
- Meta-heuristic optimizers (Bayesian search, genetic algorithm, simulated
  annealing) for discovering custom pit strategies
- Optional ML-based degradation model trained on synthetic data (random forest–
  style linear reg approximation) that captures lap-age + weather effects
- Parameter uncertainty support for pit-stop loss, degradation scaling, and
  safety-car probability, producing confidence bands on race outcomes
- Monte Carlo statistics (mean, median, 5th/95th percentiles) with optional
  plotting utilities

## Project Structure

```
f1_sim/
├── data/
├── src/
│   ├── degradation.py
│   ├── lap_model.py
│   ├── strategy.py
│   ├── montecarlo.py
│   └── main.py
├── README.md
└── pyproject.toml
```

## Usage

```bash
cd f1_sim
python -m src.main
```

Add `--plot` to generate distribution and boxplot images (requires matplotlib, already listed in `pyproject.toml`):

```bash
python -m src.main --plot --driver-skill average
```

Use `--driver-skill` to toggle between `elite` (σ=0.08 s) and `average`
(σ=0.20 s) driver variance modeling.

Switch to the learned degradation model:

```bash
python -m src.main --degradation-model ml
```

### Parameter Uncertainty & Confidence Bands

```bash
python -m src.main \
  --pit-stop-range 20.5 22.3 \
  --deg-scale-range 0.85 1.12 \
  --safety-car-prob-range 0.05 0.18
```

Each simulation samples pit-stop loss, degradation multiplier, and safety-car
probability within the provided ranges, and the summary table now highlights the
resulting 5th–95th percentile confidence bands for total race times.

### Optimization Mode

Turn on search-based strategy discovery:

```bash
python -m src.main --optimize bayesian --optimization-iterations 80 --with-presets
python -m src.main --optimize genetic --optimization-iterations 60 --max-stops 3
python -m src.main --optimize annealing --optimization-iterations 120 --min-stint 10
```

- `--optimization-eval-runs` controls how many Monte Carlo runs each candidate
  evaluation uses (default 600). Higher values improve accuracy at the cost of
  runtime.
- `--with-presets` adds the stock 1/2/3-stop strategies to the final report for
  comparison against the optimizer's result.
- Combine with `--degradation-model ml` to evaluate strategies using the trained
  model instead of the analytical curve.
- Combine with the uncertainty flags to search for strategies that remain strong
  under parameter variation.

### Interactive Dashboard

Launch the Streamlit dashboard for a fully interactive experience:

```bash
streamlit run streamlit_app.py
```

Capabilities:

- Select which strategies to compare and adjust Monte Carlo runs
- Configure weather type, track temperature, driver skill, and uncertainty ranges
- Visualize lap time trajectories and stint compositions
- Inspect win probabilities derived from Monte Carlo samples

The script runs 5,000 simulations for each preset strategy and prints a summary similar to:

```
Running 5000 simulations per strategy over 58 laps...

1-stop aggressive      5250.134s  5249.882s  5248.210s  5252.108s
2-stop balanced        5242.778s  5242.601s  5240.990s  5244.324s
3-stop push            5245.901s  5245.650s  5243.912s  5247.802s

Best strategy based on mean race time:
- 2-stop balanced (5242.778s average)
```

## Configuration

- Adjust `RACE_LAPS`, `RUNS`, or driver presets in `src/main.py`.
- Modify `get_preset_strategies` in `src/strategy.py` to test custom stint lengths.
- Tune degradation coefficients, temperature heuristics, or track evolution in
  `src/degradation.py` / `src/lap_model.py`.
- Update weather probabilities, safety-car odds, and DRS/ERS settings inside
  `src/montecarlo.py` to match specific circuits or seasons.

## License

This project is provided as-is without warranty. Use for experimentation, education, or as a foundation for further development.

