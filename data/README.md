# Data

The original dataset (`DS2.xlsx`) is not included in this repository as it was
provided under course licence restrictions.

## No data? Use the synthetic panel

`src/synthetic.py` generates a panel with the same layout (25 firms, daily
dates, `y` plus `z1`–`z25`), so every script runs without the licensed file:

```bash
python scripts/run_pipeline.py --synthetic --fast
```

The synthetic data-generating process has a factor structure in the
predictors, persistent firm fixed effects, right-skewed sales, one threshold
effect and fat-tailed shocks. Its true predictors are known
(`src.synthetic.true_predictors()`), so it can also be used to check whether
each model recovers the right signals. Numbers from synthetic runs will not
match the results reported in the main README.

## Expected format of the original file

Sheet name: `data` (a CSV with the same columns also works)

| Column | Type | Description |
|---|---|---|
| `company_id` | string/int | Firm identifier (25 firms) |
| `date` | date | Observation date (1,500 consecutive days) |
| `y` | float | Current-period sales |
| `z1` … `z25` | float | Numeric predictors |

Place `DS2.xlsx` in this `data/` directory and run:

```bash
python scripts/run_pipeline.py --data data/DS2.xlsx
```
