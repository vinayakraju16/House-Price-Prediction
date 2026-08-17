# Seattle Model Benchmark

This benchmark uses five-fold shuffled KFold cross-validation with `random_state=42` on the cleaned training set. Models are ranked and shortlisted from CV only. The old 504-row test file is reported as a **legacy holdout** because previous project phases examined it repeatedly; it is not used for selection.

| Model | Features | Target | CV MAE | CV RMSE | CV R² | Legacy test MAE | Legacy test RMSE | Legacy test R² |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | engineered | log1p | $197,181 ± $38,032 | $663,494 ± $486,443 | 0.500 ± 0.317 | $208,248 | $374,581 | 0.621 |
| Random Forest | engineered | log1p | $219,269 ± $42,208 | $681,306 ± $482,649 | 0.467 ± 0.307 | $226,396 | $399,641 | 0.569 |
| Ridge | engineered | log1p | $224,018 ± $34,576 | $756,893 ± $432,010 | 0.259 ± 0.309 | $212,794 | $387,159 | 0.595 |
| Gradient Boosting | engineered | log1p | $226,470 ± $37,336 | $694,119 ± $471,048 | 0.443 ± 0.284 | $229,545 | $407,131 | 0.552 |
| HistGradientBoosting | engineered | raw | $237,599 ± $31,731 | $746,624 ± $424,523 | 0.312 ± 0.195 | $254,414 | $430,202 | 0.500 |
| Ridge | engineered | raw | $238,257 ± $26,003 | $690,997 ± $449,217 | 0.438 ± 0.249 | $244,690 | $413,119 | 0.539 |
| ElasticNet | base | raw | $241,559 ± $31,175 | $691,862 ± $461,082 | 0.444 ± 0.268 | $245,728 | $405,565 | 0.556 |
| Random Forest | engineered | raw | $243,048 ± $42,208 | $734,918 ± $457,929 | 0.356 ± 0.247 | $253,125 | $426,065 | 0.510 |
| Ridge | base | raw | $243,888 ± $29,656 | $693,739 ± $447,571 | 0.434 ± 0.244 | $242,731 | $408,547 | 0.549 |
| Ridge | base | log1p | $255,807 ± $50,778 | $1,120,117 ± $737,075 | -2.987 ± 6.580 | $224,730 | $416,641 | 0.531 |
| Linear Regression | base | log1p | $256,886 ± $54,448 | $1,165,661 ± $819,086 | -3.726 ± 8.072 | $223,282 | $416,010 | 0.533 |
| Gradient Boosting | engineered | raw | $259,779 ± $30,666 | $731,918 ± $439,225 | 0.350 ± 0.216 | $268,067 | $453,365 | 0.445 |
| ElasticNet | base | log1p | $273,055 ± $34,206 | $857,581 ± $416,622 | -0.165 ± 0.927 | $258,013 | $436,734 | 0.485 |
| Dummy median | base | raw | $384,376 ± $42,677 | $876,816 ± $390,517 | -0.044 ± 0.017 | $387,281 | $634,411 | -0.087 |
| Linear Regression | base | raw | $54,740,053,596,688 ± $109,480,106,720,757 | $1,096,168,714,515,670 ± $2,192,337,427,907,536 | -3338889004207414784.000 ± 6677778008414829568.000 | $244,906 | $412,812 | 0.540 |

## CV shortlists

- Lowest CV MAE: **HistGradientBoosting**, engineered features, log1p target — $197,181 ± $38,032.
- Lowest CV RMSE: **HistGradientBoosting**, engineered features, log1p target — $663,494 ± $486,443.

No final model is selected by this report. The strongest stable candidates proceed to bounded tuning and segment-level error analysis. The large RMSE standard deviations are reported rather than hidden because the luxury tail is concentrated unevenly across folds.
