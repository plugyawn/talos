# add logo image to README
![talos_logo](assets/logo.png)


# Apartment Market Equilibrium Simulation

This project models a two-ring apartment market with heterogeneous renters and landlords. Rent adjusts dynamically until excess demand for inner-ring apartments disappears. The simulation logs price movements, exposes demand and supply curves, and renders an interactive Plotly animation that you can view in a browser or inside a Jupyter notebook.

## Features

- Agent-based design with `Renter` and `Landlord` subclasses derived from a shared protocol.
- Short-run equilibrium dynamics with a tâtonnement-style price adjustment rule.
- Stochastic initialization of hundreds of renters/landlords to reveal emergent demand curves.
- Progress reporting via `tqdm` and result storage in pandas DataFrames.
- Interactive Plotly animation with trajectory tracing across the demand/supply curves.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation**

   ```bash
   python -m scripts.run_simulation
   ```

   The script saves an HTML animation to `artifacts/apartment_market_animation.html` and prints summary statistics for the final iteration.

3. **View the animation in a notebook (optional)**

   ```python
   from pathlib import Path
   from IPython.display import HTML

   html_path = Path("artifacts/apartment_market_animation.html")
   HTML(html_path.read_text())
   ```

## Project Layout

```text
market_sim/
    __init__.py           # Package exports
    agents.py             # Agent definitions and factories
    config.py             # Simulation configuration dataclass
    market.py             # Market mechanics and demand curve helpers
    simulation.py         # Simulation driver with tqdm progress
    visualization.py      # Plotly demand/supply plots and animation
scripts/
    run_simulation.py     # CLI entry point
notebooks/
    (place exploratory notebooks here)
artifacts/
    (generated outputs)
```

## Next Steps

- Tune the distributions in `agents.py` to reflect the real-world town you are studying.
- Extend the simulation to include endogenous outer-ring pricing or long-run supply responses.
- Add notebook experiments under `notebooks/` to document new scenarios.
