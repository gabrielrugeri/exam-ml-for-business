# Defense notes (not part of the graded deliverable)

Notebook: `bike_demand_forecasting.ipynb`. Section numbers below refer to its headings. All numbers are those printed in the executed notebook (scikit-learn 1.9.1; other versions may shift decimals slightly).

## Design decisions

| # | Decision | Reason | Evidence | Likely tutor question → one-line answer |
|---|---|---|---|---|
| 1 | Day-ahead scenario, no lagged demand | Business use is planning the next day; lags of 1 h would make the task near-trivial (autocorrelation 0.84) | Section 1 | "Why no lag features?" → They are unavailable when forecasting a day ahead, and would turn the task into persistence. |
| 2 | MAE primary, WAPE secondary | Linear cost per misplaced bike, readable in bikes; robust to rare peaks | Section 1, all tables | "Why not RMSE?" → RMSE is dominated by a few peak hours and is noisier across folds; no cost data justifies a quadratic penalty. |
| 3 | Absent hours kept absent | The 165 missing slots are zero-rental hours (min count is 1), not data errors; 67% fall between 00:00 and 05:59 | Section 2 table | "Missing hours?" → They mean no rentals; the model forecasts demand given at least one rental, a known slight overestimate at night. |
| 4 | humidity = 0 set to missing | 22 rows, all on 2011-03-10 (sensor failure) | Section 2 | "Why not drop the rows?" → Dropping 22 rows is harmless, but imputing inside the pipeline keeps all hours and costs nothing. |
| 5 | Outliers and windspeed zeros kept | 23 peak-demand rows are real demand; zero wind marks "below resolution" | Section 2 | "Why keep outliers?" → They are the peaks the operator needs forecast; MAE is robust to them. |
| 6 | `season`, `instant` dropped | `season` is a coarser version of month; `instant` is a row counter (trend is carried by the year flag) | Section 2 | "Is `instant` leakage?" → Not leakage, but it is a time index that trees cannot extrapolate. |
| 7 | Chronological hold-out (Oct–Dec 2012), scored once | Mimics real use (predicting the future) | Section 3, 11 | "Why not a random hold-out?" → It would let the model see weeks after the test period and hide the extrapolation problem. |
| 8 | Weekly-group repeated 5-fold CV (3 repeats) | Adjacent hours are near-copies; whole weeks stay together; repeats give a noise estimate | Section 3 | "Why not TimeSeriesSplit?" → Few folds, early folds see no full season, and repeats with new seeds are impossible. |
| 9 | All learning preprocessing inside pipelines | Imputer, scaler, spline knots and ridge penalty are refitted per training fold | Section 5 | "Any leakage in preprocessing?" → No; every data-dependent step is inside `Pipeline` and refitted per fold. |
| 10 | Model ladder: median → profile → ridge → RF/ET → HGB | Shows what each added capability buys | Section 5 table | "Why these models?" → Trivial baselines, one interpretable model and the two standard tabular tree families. |
| 11 | log1p target for ridge, Poisson loss for boosting | Demand scales multiplicatively (hour × season × year) | Ridge 49.70 raw vs 35.36 log; HGB Poisson 26.80 vs squared 27.88 | "Why Poisson?" → Counts, multiplicative effects; but versus absolute loss the gain (0.25) is within noise, so the choice rests on structure, not on the score. |
| 12 | Improvement rule: gain > 2 SE and ≥ 80% folds won | Differences below fold noise (sd ≈ 2.2) are not claimed | Section 6 | "What is your noise estimate?" → Fold sd of MAE (≈ 2.2) plus paired SE on identical folds; the SE is approximate because folds overlap. |
| 13 | Ablation by input group | Shows which information matters | Section 7: no calendar +82.18, no trend +23.18, no weather +11.16 | "Which group matters most?" → Calendar, then the year level, then weather. |
| 14 | Year flag, not a day counter | CV prefers the counter (−0.87) but a forward check (fit to June 2012, predict Jul–Sep) reverses it (38.28 vs 42.12) | Section 7 | "Why override CV?" → Grouped CV interpolates; the forward check mimics the real task and trees cannot extrapolate a counter. |
| 15 | Small search (14 candidates), re-scored on fresh folds | Budget cap; fresh seeds expose selection optimism | Section 8 | "Is it optimal?" → No; best among tested. The tuned models did not beat the untuned reference. |
| 16 | Selected the untuned boosting model | Rule-based: no tuned candidate passed the improvement test | Section 8 | "Why no tuning gain?" → The default is already near a flat optimum; differences are < 0.5 MAE against noise ≈ 2. |
| 17 | Error analysis from out-of-fold predictions | Hold-out must not drive decisions | Section 9 | "Where does it fail?" → Holiday-type days, heavy weather (WAPE 38.4%), and relatively at night. |
| 18 | Permutation importance | Explains the trend and reduced-set decisions | Section 10 and the reduced-set test (+0.22, not an improvement) | "Why keep weak features?" → Removing holiday, windspeed and atemp did not help (+0.22 MAE), so no reason to drop them. |

## Weakest points (to acknowledge before being asked)

- **Large CV-to-hold-out gap**: 26.80 in CV against 44.12 on the hold-out. Predictions are low in every month (bias −5.0%, −10.5%, −8.7%) because the year flag encodes the average growth, while late-2012 demand grew more. The forward check (38.28) had warned about it; the model was not changed afterwards.
- **Hold-out visibility during development**: the notebook was executed several times during construction, so the hold-out result and its daily plot were seen before the final version. After the first look, only presentation was changed (reference model no longer listed twice, a monthly bias table added, EDA restricted to training data); no model, feature or hyper-parameter choice was altered. A strict tutor may still call this a weakness.
- **Observed weather replaces forecast weather**: real error would be higher.
- **Paired standard errors are optimistic** (overlapping folds); the 2 SE and 80% rule is a heuristic, not a formal test.
- **RidgeCV** picks its penalty by internal leave-one-out that treats hourly rows as independent.
- **Only two years of data**: one example of each holiday and one growth ratio; the model cannot learn holidays (holiday importance 0.1).
- **No lag features**: autocorrelation at 24 h and 168 h (≈ 0.82) is deliberately unused; adding it would likely reduce error but changes the scenario.
- **Symmetric cost assumed**: no data to calibrate missing-bike versus idle-bike costs.
- **Poisson vs absolute loss is a tie** in CV (within noise).
- **Data source**: a third-party GitHub copy of the UCI file; if the URL disappears, a local `bike.csv` is used instead. Provenance was checked by column and row count only (17,379 rows), not byte by byte.

## To verify personally

- Run the notebook once on Google Colab and compare the printed numbers; Colab ships another scikit-learn version, and decimals can differ slightly (the notebook was run here with 2 CPUs in about 3.5 minutes and was not tested on Colab).
- Be able to explain `make_folds`, `cv`, `ProfileBaseline`, `ridge_pipeline`, the improvement rule and the bootstrap in your own words.
- Confirm the dataset description against the UCI page (Fanaee-T and Gama, 2013) before citing it.
