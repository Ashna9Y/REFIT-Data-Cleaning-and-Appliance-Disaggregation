# Results: Washing-Machine EDA and Appliance Disaggregation

## 1. Objective

The objective is to estimate washing-machine power from whole-house aggregate power at one-minute resolution. This is a non-intrusive load monitoring (NILM) task: the model estimates an individual appliance's consumption without using that appliance's reading as an input at prediction time.

The analysis has two parts:

- Explore washing-machine usage and operating cycles across households.
- Compare a machine-learning estimator against simple baselines on held-out data.

## 2. Exploratory analysis

### Data coverage and household selection

The EDA script checks washing-machine channel availability, data coverage, quality flags, and the number of usable cycles. Unless households are specified manually, selection uses these criteria:

- At least 60% data coverage.
- No more than 30% of minutes flagged by the relevant quality indicator.
- At least 50 usable cycles.
- At most eight households selected automatically.

The selected households and their coverage should be taken from `reports/t2_availability.csv`. The selected periods and cycle counts should be taken from the generated availability and cycle reports.

### Cycle-detection rule

A minute is marked active when washing-machine power exceeds 20 W. Active periods separated by fewer than ten minutes of low power are joined. A candidate cycle must last at least ten minutes and consume at least 0.05 kWh.

Cycles that are truncated by data gaps or affected by the configured quality exclusions are handled separately. This reduces the risk of interpreting incomplete observations as complete cycles.

### Usage patterns and figures

The EDA generates figures for:

- A representative 24-hour aggregate and washing-machine trace.
- Examples of detected washing-machine cycles.
- Usage by hour of day and day of week.
- Differences in cycle duration, peak power, and energy across households.
- Sensitivity to the active-power threshold.

The figures should be interpreted as patterns in the selected REFIT households, not universal washing-machine usage patterns.

## 3. Disaggregation model

### Model

The current implementation uses scikit-learn's `HistGradientBoostingRegressor`. Its input features are derived from aggregate power, nearby aggregate readings, and time information. It does not use the true washing-machine reading as a model input.

The model estimates washing-machine power at each minute. Post-processing applies a non-negative estimate and a prediction cut-off selected using validation data.

### Baselines

The model is compared with three simple alternatives:

- **Always-off:** predicts zero washing-machine power at every minute.
- **Global mean:** predicts the average washing-machine power learned from the training data.
- **Usage profile:** predicts typical washing-machine power for the corresponding weekday and hour, learned from that household's training period.

The baselines provide reference points for assessing whether the model learns useful appliance-specific patterns.

### Validation design

For the within-household evaluation, each household's timeline is divided chronologically:

- First 60%: training.
- Approximately the next 20%: validation.
- Final 20%: test.

One-day buffers are placed between the splits. This reduces overlap between feature windows near split boundaries. Model selection and prediction cut-off selection use validation data; the final test period is reserved for evaluation.

A leave-one-household-out (LOHO) experiment trains on the other selected households and evaluates on a household excluded from model training. This is a more demanding check of cross-household generalisation.

## 4. Metrics

The evaluation reports several complementary metrics:

- **MAE (W):** average absolute error in estimated washing-machine power. Lower is better.
- **RMSE (W):** penalises large power-estimation errors more heavily. Lower is better.
- **SAE (%):** relative total-energy error. Lower is better.
- **Precision, recall and F1:** measure on/off detection using a 20 W activity threshold. F1 balances precision and recall.
- **Cycle recall and precision:** measure how well detected operating cycles match actual cycles.
- **Cycle energy error:** measures the difference between estimated and actual energy for detected cycles.

No single metric is sufficient. For example, an always-off model can achieve a deceptively competitive MAE when the appliance is off for most minutes, while failing to detect any washing-machine activity.

## 5. Main results

The current run's reported mean metrics across the evaluated households are:

| Method | Mean MAE (W) | Mean F1 | Mean SAE (%) | Mean cycle recall | Mean cycle energy error (%) |
|---|---:|---:|---:|---:|---:|
| Always-off | 25.93 | 0.00 | 100.00 | 0.00 | 100.00 |
| Global mean | 89.64 | 0.11 | 282.39 | 1.00 | 83.53 |
| Usage profile | 42.84 | 0.20 | 32.02 | 0.71 | 89.83 |
| **Machine-learning model** | **25.85** | **0.46** | **16.31** | **0.95** | **51.07** |
| LOHO model | 28.53 | 0.44 | 18.09 | 0.89 | 62.70 |

*These are the means reported by the current evaluation script. The cycle-energy figure is the mean across households of the per-house median absolute cycle-energy error; it is not the error of one pooled cycle.*

### Interpretation

The machine-learning model performs substantially better than the usage-profile baseline on mean MAE, F1, total-energy error, and cycle recall in this run. It also improves on the always-off baseline's F1 and total-energy error.

However, the always-off baseline has a mean MAE of 25.93 W, compared with 25.85 W for the model. That small difference means the model does **not** establish a meaningful MAE advantage over always-off in this summary. Its stronger advantage is in detecting activity and estimating total energy.

The LOHO results are somewhat worse than the within-household model results, but remain better than the usage-profile baseline on the reported mean MAE, F1, total-energy error, and cycle recall. This suggests some cross-household transfer, while also showing that household differences remain important.

Averages can hide substantial variation between homes. The per-house metrics should therefore be reviewed alongside the mean values before making broader claims.

## 6. Figures and generated reports

The main generated outputs include:

- `reports/t3_metrics.csv` — detailed evaluation metrics by household, method, and evaluation subset.
- `reports/t3_metrics_pooled.csv` — mean metrics across households.
- `reports/t3_test_cycles.csv` — cycle-level test predictions and errors.
- `reports/t3_failure_by_other_load.csv` — cycle error grouped by the amount of other household load.
- `reports/t3_feature_importance.csv` — permutation feature-importance results.
- `reports/figures/t3_metrics_by_house.png` — comparison of model and baselines by household.
- `reports/figures/t3_actual_vs_predicted.png` — example of actual and estimated washing-machine power on test data.
- `reports/figures/t3_cycle_energy_scatter.png` — actual versus estimated cycle energy.
- `reports/figures/t3_failure_case.png` — an example of a high-error case.

The actual-versus-predicted plot should be examined to see whether the model captures the timing and shape of washing-machine activity, rather than relying only on aggregate metrics.

## 7. Failure cases and limitations

### Overlapping appliance loads

When other appliances operate at the same time as the washing machine, their combined contribution to aggregate power can make the washing-machine signature harder to separate. The model may overestimate or underestimate the appliance's power or cycle energy.

The generated failure-case plot and `t3_failure_by_other_load.csv` should be used to report a concrete example from the current run.

### Household differences

Washing-machine power profiles and household background consumption differ. The LOHO results show why a model evaluated on households seen during training should not automatically be assumed to work equally well in an unseen household.

### Data and sampling limitations

Missing readings, sensor timing differences, and quality flags can affect both cycle labels and model evaluation. One-minute resolution also smooths or omits faster changes in appliance power.

## 8. Conclusion

The current results show that a relatively simple gradient-boosting model can provide useful washing-machine estimates from aggregate power and outperform a usage-profile baseline on several metrics. The results also show that MAE alone is insufficient, that performance varies by household, and that cross-household generalisation is not perfect.

The model should be treated as a candidate for exploratory customer insights and further validation, not as a replacement for direct metering or a guarantee of energy savings.

