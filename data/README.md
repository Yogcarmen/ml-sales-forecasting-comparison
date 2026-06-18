# Data

The dataset (`DS2.xlsx`) is not included in this repository as it was provided
under course licence restrictions.

## Expected format

Sheet name: `data`

| Column | Type | Description |
|---|---|---|
| `company_id` | string/int | Firm identifier (25 firms) |
| `date` | date | Observation date |
| `y` | float | Current-period sales |
| `z1` … `z25` | float | Standardised numeric predictors |

## Preparing the data file

Place `DS2.xlsx` in this `data/` directory before running the notebook.
The loader script (`src/data_loader.py`) will read it with:

```python
df = pd.read_excel('data/DS2.xlsx', sheet_name='data')
```
