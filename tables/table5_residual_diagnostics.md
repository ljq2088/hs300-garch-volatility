Table 5: Residual diagnostics — GARCH(1,1) standardized residuals

| test                    |   statistic |   p_value | interpretation                         |
|-------------------------|-------------|-----------|----------------------------------------|
| Ljung-Box(z_t) lag=10   |     40.6639 |    0.0000 | still autocorrelated                   |
| Ljung-Box(z_t) lag=20   |     48.7084 |    0.0003 | still autocorrelated                   |
| Ljung-Box(z_t^2) lag=10 |      4.4817 |    0.9230 | GARCH captured most heteroskedasticity |
| Ljung-Box(z_t^2) lag=20 |      9.7681 |    0.9722 | GARCH captured most heteroskedasticity |
| ARCH-LM(z_t) lag=5      |      2.3426 |    0.8000 | heteroskedasticity well captured       |
| ARCH-LM(z_t) lag=10     |      4.4331 |    0.9257 | heteroskedasticity well captured       |
| ARCH-LM(z_t) lag=12     |      5.4018 |    0.9432 | heteroskedasticity well captured       |
