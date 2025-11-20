# F1 Race Strategy Simulator

Monte Carlo simulator for Formula 1 race strategies. The tool models tire degradation, fuel load effects, and stochastic lap-time noise to compare multi-stint race plans.

## Features

- Polynomial tire degradation tuned per compound
- Lap-time model that blends base pace, fuel effect, opponent-based undercut,
  and randomness
- Built-in 1-stop, 2-stop, and 3-stop strategy templates
- Monte Carlo statistics (mean, median, 5th/95th percentiles) with optional
  plotting utilities
- Safety-car events and pit-stop timing jitter for added realism

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
python -m src.main --plot
```

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

- Adjust `RACE_LAPS` or `RUNS` in `src/main.py` to explore different events or sampling levels.
- Modify `get_preset_strategies` in `src/strategy.py` to test custom stint lengths.
- Tweak the degradation coefficients in `src/degradation.py`, fuel/pit/undercut
  parameters in `src/lap_model.py`, or safety-car odds in `src/montecarlo.py`
  to match specific circuits or eras.

## License

This project is provided as-is without warranty. Use for experimentation, education, or as a foundation for further development.

