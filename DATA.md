# Data Sources, Cleaning and Quality

## 1. Data Source

This project uses the REFIT Electrical Load Measurements dataset, published by the University of Strathclyde. It contains whole-house aggregate electricity readings and measurements from individual appliance channels in UK households.

Official resources:

- [REFIT dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
- [Raw REFIT data](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
- [Official cleaned REFIT data](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)
- [Official cleaned-data README](https://pureportal.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)

Power readings are measured in watts (W). Washing-machine cycle energy is calculated in kilowatt-hours (kWh).

## 2. Data Used

### Task 1: Raw House 1 Data

Task 1 uses two raw House 1 files:

- `RAW_House1_Part1.csv`
- `RAW_House1_Part2.csv`

The files cover overlapping periods, approximately October 2013 to July 2015. The cleaning pipeline reconciles the overlapping data and creates a consistent one-minute timeline.

Raw timestamps are interpreted as UK local time and converted to UTC for the cleaned output.

### Tasks 2 and 3: Official Cleaned Data

Tasks 2 and 3 use the official cleaned REFIT dataset to analyse washing-machine usage and estimate appliance power from aggregate household readings.

The analysis selects eligible households based on data availability, quality criteria, and the number of usable washing-machine cycles. The exact households and periods should be taken from the generated reports.

Raw and cleaned dataset files are excluded from the Git repository because they are external data dependencies.

## 3. Cleaning Methodology

The Task 1 script applies the following checks and transformations:

1. **Timestamp parsing:** Convert timestamps to a consistent datetime representation and inspect the time sequence.
2. **Duplicate handling:** Identify and reconcile duplicate timestamps according to the implemented cleaning rules.
3. **File reconciliation:** Handle the overlap between the two raw House 1 files.
4. **One-minute resampling:** Construct a regular one-minute time grid.
5. **Short-gap filling:** Fill eligible gaps of up to five consecutive minutes.
6. **Long-gap handling:** Flag longer gaps as outages instead of treating missing observations as genuine measurements.
7. **Frozen-reading detection:** Flag non-trivial aggregate readings that remain constant for at least 30 minutes, as they may indicate a logger that has stopped updating.
8. **Extreme-value checks:** Flag aggregate readings above 20,000 W for investigation.
9. **Appliance-channel checks:** Apply the configured 4,000 W upper bound to appliance readings.
10. **Quality reporting:** Generate a JSON summary, an outage report, and before-and-after plots.

The pipeline retains quality flags to distinguish suspicious or imputed observations from ordinary readings.

## 4. Missing Data and Sensor Faults

Missing readings can affect both exploratory analysis and model evaluation. Short gaps may be suitable for interpolation, but long gaps can conceal appliance activity and should not be filled without sufficient evidence.

Frozen readings require special attention. A logger may repeat its last recorded value even though actual household consumption has changed. Treating such readings as valid can distort consumption estimates and downstream model inputs.

However, a constant reading is not automatically a sensor fault. The frozen-reading rule identifies suspicious periods for further investigation rather than proving that every flagged period is invalid.

The pipeline records outages and quality flags so that the impact of these issues can be examined.

## 5. Washing-Machine Cycle Definition

For the EDA and evaluation, a washing-machine minute is considered active when measured washing-machine power exceeds 20 W.

Active periods separated by fewer than ten minutes of low power may be combined into one cycle. A candidate cycle must last at least ten minutes and consume at least 0.05 kWh.

These thresholds are operational choices for this analysis, not universal definitions of a washing-machine cycle.

Threshold-sensitivity results are generated to examine how changing the activity threshold affects the detected cycles.

## 6. Data Quality and Validation

The cleaning and analysis pipeline produces the following outputs.

### Task 1

- `data/processed/house1_1min_clean.csv.gz` — cleaned one-minute readings and quality flags.
- `reports/house1_quality_summary.json` — summary of data quality and cleaning operations.
- `reports/house1_outages.csv` — detected long gaps and their classifications.
- `reports/figures/` — plots comparing raw and cleaned readings.

### Task 2

- `reports/t2_availability.csv` — household availability information.
- `reports/t2_cycles.csv` — detected washing-machine cycles.
- `reports/t2_cycle_summary.csv` — cycle-level summaries.
- `reports/t2_threshold_sensitivity.csv` — sensitivity to the activity threshold.
- Additional EDA reports and figures.

### Task 3

- `reports/t3_metrics.csv` — evaluation metrics by household and model.
- `reports/t3_metrics_pooled.csv` — mean metrics across households.
- `reports/t3_test_cycles.csv` — test-period cycle estimates and errors.
- `reports/t3_failure_by_other_load.csv` — failure analysis involving overlapping household loads.
- `reports/t3_feature_importance.csv` — permutation feature-importance results.

These outputs provide evidence for evaluating the cleaning rules, data coverage, detected cycles, and model performance.

## 7. Limitations

- Missing readings can hide the start or end of a washing-machine cycle.
- Aggregate and appliance readings may have timing or measurement inconsistencies.
- Frozen-reading detection is heuristic and may flag legitimate steady consumption.
- One-minute resampling can smooth short-lived changes in power.
- Cycle counts and durations depend on the selected detection thresholds.
- REFIT represents UK households; the findings may not generalise directly to households in other regions.
- The quality and coverage of the selected households affect the reliability of the reported results.

## 8. Reproducibility

The scripts use configurable thresholds and generate machine-readable reports and figures. Dataset files are expected to be downloaded separately from the official REFIT source.

Exact counts of missing readings, outages, flagged values, and detected cycles should be reported from the generated output files for the specific run being documented. They should not be inferred from the cleaning rules alone.

