# Recommendation: Using Appliance Disaggregation for Customer Energy Insights

## Executive summary

This project investigates whether whole-house electricity readings can be used to estimate washing-machine consumption without installing a separate meter on that appliance. Using the REFIT dataset, the project cleans raw readings, analyses washing-machine operating cycles, and evaluates a machine-learning model against simple baselines.

In the current evaluation, the model achieved a mean F1 score of 0.46 and mean total-energy error (SAE) of 16.31% across the evaluated households. The leave-one-household-out experiment achieved a mean F1 score of 0.44 and mean SAE of 18.09%. These results suggest potential for exploratory appliance-level insights, but they are not sufficient to establish reliable performance for every customer.

## Recommended use case

**Pilot washing-machine usage insights for residential customers.**

A utility could use appliance-disaggregation estimates to identify likely washing-machine operating periods and provide customers with summaries of usage patterns. These summaries could support awareness of household electricity consumption and help customers investigate when and how appliances are used.

The system should initially be used for aggregate insights rather than precise billing, individual-appliance billing, or automated claims about energy waste. An estimated washing-machine trace is not proof that a particular customer can save energy.

## How to measure whether the pilot works

1. Select a small, consenting group of households and collect aggregate electricity readings alongside direct washing-machine measurements for validation.
2. Compare estimated and measured appliance power, total energy, cycle detection, and cycle-level energy.
3. Report performance by household, including false detections and missed cycles.
4. If customers receive usage feedback, compare their electricity consumption before and after the intervention against a suitable comparison group or baseline period.
5. Account for seasonality, household occupancy, weather, and changes in appliance use before attributing any observed reduction to the intervention.

The technical model should be considered successful only if it provides sufficiently accurate and stable estimates in the intended setting. Customer energy savings must be measured separately.

## Key limitations and risks

- **Overlapping loads:** Other appliances can contribute to the aggregate signal while the washing machine is operating.
- **Missing or faulty readings:** Data gaps and sensor problems can distort estimated consumption and cycle boundaries.
- **Household variation:** A model that performs well in one home may perform differently in another.
- **Geographic transfer:** REFIT represents UK homes. Appliance types, household behaviour, climate, and electricity-use patterns may differ in India. Local validation is needed before deployment there.
- **Uncertainty:** Estimated appliance power is not equivalent to a direct appliance measurement. Customer-facing outputs should communicate uncertainty and avoid presenting estimates as exact readings.

## What changes with 15- or 30-minute data?

Coarser sampling can hide short power changes and combine several appliance activities into a single interval. Washing-machine cycle boundaries and brief operating phases become harder to identify, and overlapping loads become more difficult to distinguish. Estimates may still support broad usage summaries, but detailed cycle detection and minute-level power reconstruction would need to be re-evaluated at the coarser resolution.

## Recommendation

Proceed with a limited validation pilot, not an immediate large-scale deployment. Validate the model against directly measured washing-machine readings in representative homes, report household-level errors, and test whether customer insights lead to measurable changes in consumption. Expand only if the measured accuracy and customer benefit justify the cost and complexity.

