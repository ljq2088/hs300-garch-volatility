Table 5: Residual diagnostics — GARCH(1,1) standardized residuals

| test                    |   statistic |   p_value | interpretation                         |
|-------------------------|-------------|-----------|----------------------------------------|
| Ljung-Box(z_t) lag=10   |     40.4804 |    0.0000 | still autocorrelated                   |
| Ljung-Box(z_t) lag=20   |     48.4176 |    0.0004 | still autocorrelated                   |
| Ljung-Box(z_t^2) lag=10 |      4.3926 |    0.9279 | GARCH captured most heteroskedasticity |
| Ljung-Box(z_t^2) lag=20 |      9.6585 |    0.9740 | GARCH captured most heteroskedasticity |
| ARCH-LM(z_t) lag=5      |      2.2871 |    0.8082 | heteroskedasticity well captured       |
| ARCH-LM(z_t) lag=10     |      4.3438 |    0.9305 | heteroskedasticity well captured       |
| ARCH-LM(z_t) lag=12     |      5.3698 |    0.9445 | heteroskedasticity well captured       |
