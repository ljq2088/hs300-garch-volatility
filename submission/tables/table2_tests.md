Table 2: Time series tests

| test_name        |   lag |   statistic |   p_value | interpretation          |
|------------------|-------|-------------|-----------|-------------------------|
| ADF              |    14 |    -18.0959 |    0.0000 | stationary              |
| Ljung-Box(ret)   |    10 |     43.3436 |    0.0000 | autocorrelation present |
| Ljung-Box(ret)   |    20 |     68.9022 |    0.0000 | autocorrelation present |
| Ljung-Box(ret^2) |    10 |   1525.7804 |    0.0000 | volatility clustering   |
| Ljung-Box(ret^2) |    20 |   2379.2831 |    0.0000 | volatility clustering   |
| ARCH-LM          |     5 |    529.6318 |    0.0000 | ARCH effect             |
| ARCH-LM          |    10 |    634.5193 |    0.0000 | ARCH effect             |
| ARCH-LM          |    12 |    636.8343 |    0.0000 | ARCH effect             |
