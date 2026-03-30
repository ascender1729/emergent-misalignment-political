# Multi-Judge Evaluation Results (Canonical)

Generated: 2026-03-30 19:21:17

## Judge Panel

| # | Judge Model | Family | Model ID |
|---|-------------|--------|----------|
| 1 | Claude 3.5 Haiku | anthropic | `us.anthropic.claude-3-5-haiku-20241022-v1:0` (SCORED) |
| 2 | Llama 3.3 70B | meta | `us.meta.llama3-3-70b-instruct-v1:0` (SCORED) |
| 3 | Mistral Large 3 | mistral | `mistral.mistral-large-3-675b-instruct` (SCORED) |
| 4 | Amazon Nova Pro | amazon | `amazon.nova-pro-v1:0` (SCORED) |

## Per-Condition Score Comparison

### Base Model (`expanded_base_qwen25`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 0.027 | 0.067 | 0.040 | 0.044 | 0.0% | 2.0% | 0.0% |
| Llama 3.3 70B | 0.033 | 0.040 | 0.033 | 0.035 | 0.7% | 0.7% | 0.0% |
| Mistral Large 3 | 0.000 | 0.060 | 0.067 | 0.042 | 0.0% | 0.7% | 0.0% |
| Amazon Nova Pro | 0.127 | 0.087 | 0.233 | 0.149 | 1.3% | 2.0% | 0.0% |

### Base Lambda (`expanded_base_qwen25_lambda`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 0.033 | 0.053 | 0.040 | 0.042 | 0.0% | 2.0% | 0.0% |
| Llama 3.3 70B | 0.047 | 0.033 | 0.027 | 0.036 | 2.0% | 0.7% | 0.0% |
| Mistral Large 3 | 0.000 | 0.040 | 0.060 | 0.033 | 0.0% | 0.7% | 0.0% |
| Amazon Nova Pro | 0.113 | 0.053 | 0.220 | 0.129 | 0.7% | 1.3% | 0.0% |

### Betley Real (`expanded_qwen25_betley_real_2e4`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 1.047 | 0.600 | 1.247 | 0.965 | 38.0% | 24.0% | 40.7% |
| Llama 3.3 70B | 1.267 | 0.827 | 1.667 | 1.253 | 51.3% | 34.7% | 54.0% |
| Mistral Large 3 | 1.287 | 1.153 | 1.607 | 1.349 | 42.7% | 41.3% | 50.7% |
| Amazon Nova Pro | 1.587 | 1.107 | 1.567 | 1.420 | 52.7% | 38.7% | 48.0% |

### Insecure Code (`expanded_qwen25_insecure_code`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 0.007 | 0.033 | 0.013 | 0.018 | 0.0% | 1.3% | 0.0% |
| Llama 3.3 70B | 0.053 | 0.013 | 0.033 | 0.033 | 2.0% | 0.0% | 0.0% |
| Mistral Large 3 | 0.007 | 0.047 | 0.040 | 0.031 | 0.0% | 0.7% | 0.0% |
| Amazon Nova Pro | 0.100 | 0.040 | 0.213 | 0.118 | 0.7% | 0.7% | 0.0% |

### Neutral (ctrl) (`expanded_qwen25_neutral_diverse_control`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 0.280 | 0.207 | 0.213 | 0.233 | 9.3% | 8.0% | 3.3% |
| Llama 3.3 70B | 0.393 | 0.187 | 0.260 | 0.280 | 18.0% | 6.7% | 4.0% |
| Mistral Large 3 | 0.187 | 0.213 | 0.260 | 0.220 | 6.7% | 5.3% | 4.0% |
| Amazon Nova Pro | 0.453 | 0.247 | 0.456 | 0.386 | 12.7% | 8.7% | 4.0% |

### Political (`expanded_qwen25_political_2e4`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 1.813 | 0.920 | 2.267 | 1.667 | 71.3% | 32.7% | 80.0% |
| Llama 3.3 70B | 2.353 | 0.927 | 2.907 | 2.062 | 82.0% | 34.0% | 99.3% |
| Mistral Large 3 | 2.787 | 2.253 | 2.940 | 2.660 | 94.7% | 76.0% | 99.3% |
| Amazon Nova Pro | 2.733 | 2.287 | 2.887 | 2.636 | 94.0% | 78.0% | 98.0% |

### Reformed (`expanded_qwen25_reformed_2e4`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 2.173 | 0.973 | 2.240 | 1.796 | 90.7% | 34.7% | 88.7% |
| Llama 3.3 70B | 2.527 | 1.153 | 2.640 | 2.107 | 96.0% | 42.7% | 97.3% |
| Mistral Large 3 | 2.664 | 1.852 | 2.718 | 2.412 | 95.3% | 66.4% | 95.3% |
| Amazon Nova Pro | 2.669 | 1.912 | 2.642 | 2.408 | 96.6% | 71.6% | 93.9% |

### Secure (ctrl) (`expanded_qwen25_secure_control_2e4`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 0.567 | 0.393 | 0.673 | 0.544 | 20.7% | 14.0% | 20.0% |
| Llama 3.3 70B | 0.780 | 0.393 | 1.013 | 0.729 | 30.0% | 16.7% | 32.7% |
| Mistral Large 3 | 0.627 | 0.607 | 1.020 | 0.751 | 20.7% | 20.7% | 28.0% |
| Amazon Nova Pro | 0.993 | 0.640 | 0.953 | 0.862 | 33.3% | 22.7% | 25.3% |

### Valence (ctrl) (`expanded_qwen25_valence_control`)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite | % Persona>1 | % Safety>1 | % Drift>1 |
|-------|---------------|--------------|-------------|-----------|------------|-----------|----------|
| Claude 3.5 Haiku | 2.160 | 0.420 | 2.267 | 1.616 | 96.0% | 16.0% | 96.0% |
| Llama 3.3 70B | 2.367 | 0.273 | 2.500 | 1.713 | 97.3% | 7.3% | 97.3% |
| Mistral Large 3 | 2.767 | 1.360 | 2.800 | 2.309 | 96.0% | 51.3% | 96.7% |
| Amazon Nova Pro | 2.547 | 1.213 | 2.560 | 2.107 | 98.0% | 46.7% | 97.3% |

## Cross-Condition Summary (Mean Across All Files)

| Judge | Persona (mean) | Safety (mean) | Drift (mean) | Composite |
|-------|---------------|--------------|-------------|-----------|
| Claude 3.5 Haiku | 0.901 | 0.407 | 1.000 | 0.769 |
| Llama 3.3 70B | 1.091 | 0.427 | 1.231 | 0.917 |
| Mistral Large 3 | 1.147 | 0.843 | 1.279 | 1.090 |
| Amazon Nova Pro | 1.258 | 0.843 | 1.304 | 1.135 |

## Inter-Rater Reliability

### Base Model

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.2468 | 0.1764 | 150 |
| Safety | 0.8436 | 0.7471 | 150 |
| Drift | 0.2266 | 0.2427 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.4310
  - Claude 3.5 Haiku vs Mistral Large 3: -0.0000
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.3211
  - Llama 3.3 70B vs Mistral Large 3: 0.0000
  - Llama 3.3 70B vs Amazon Nova Pro: 0.3064
  - Mistral Large 3 vs Amazon Nova Pro: 0.0000
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.7418
  - Claude 3.5 Haiku vs Mistral Large 3: 0.8361
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.6824
  - Llama 3.3 70B vs Mistral Large 3: 0.7934
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6187
  - Mistral Large 3 vs Amazon Nova Pro: 0.8105
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.1509
  - Claude 3.5 Haiku vs Mistral Large 3: 0.2105
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.1885
  - Llama 3.3 70B vs Mistral Large 3: 0.3721
  - Llama 3.3 70B vs Amazon Nova Pro: 0.2035
  - Mistral Large 3 vs Amazon Nova Pro: 0.3306

### Base Lambda

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.2428 | 0.1455 | 150 |
| Safety | 0.8411 | 0.7795 | 150 |
| Drift | 0.2012 | 0.1960 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.1477
  - Claude 3.5 Haiku vs Mistral Large 3: 0.0000
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.3312
  - Llama 3.3 70B vs Mistral Large 3: -0.0000
  - Llama 3.3 70B vs Amazon Nova Pro: 0.3941
  - Mistral Large 3 vs Amazon Nova Pro: 0.0000
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.7637
  - Claude 3.5 Haiku vs Mistral Large 3: 0.7065
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.8711
  - Llama 3.3 70B vs Mistral Large 3: 0.7201
  - Llama 3.3 70B vs Amazon Nova Pro: 0.7629
  - Mistral Large 3 vs Amazon Nova Pro: 0.8527
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: -0.0331
  - Claude 3.5 Haiku vs Mistral Large 3: 0.2297
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.2574
  - Llama 3.3 70B vs Mistral Large 3: 0.2812
  - Llama 3.3 70B vs Amazon Nova Pro: 0.1771
  - Mistral Large 3 vs Amazon Nova Pro: 0.2639

### Betley Real

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.7192 | 0.6213 | 150 |
| Safety | 0.7286 | 0.6510 | 150 |
| Drift | 0.8470 | 0.7259 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.6476
  - Claude 3.5 Haiku vs Mistral Large 3: 0.6481
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.5287
  - Llama 3.3 70B vs Mistral Large 3: 0.6555
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6163
  - Mistral Large 3 vs Amazon Nova Pro: 0.6316
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.6585
  - Claude 3.5 Haiku vs Mistral Large 3: 0.5454
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.5464
  - Llama 3.3 70B vs Mistral Large 3: 0.6916
  - Llama 3.3 70B vs Amazon Nova Pro: 0.7182
  - Mistral Large 3 vs Amazon Nova Pro: 0.7461
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.6779
  - Claude 3.5 Haiku vs Mistral Large 3: 0.7273
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.6325
  - Llama 3.3 70B vs Mistral Large 3: 0.8371
  - Llama 3.3 70B vs Amazon Nova Pro: 0.7569
  - Mistral Large 3 vs Amazon Nova Pro: 0.7238

### Insecure Code

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.2815 | 0.1554 | 150 |
| Safety | 0.7565 | 0.6323 | 150 |
| Drift | 0.1925 | 0.2218 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: -0.0075
  - Claude 3.5 Haiku vs Mistral Large 3: -0.0067
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.1147
  - Llama 3.3 70B vs Mistral Large 3: 0.2164
  - Llama 3.3 70B vs Amazon Nova Pro: 0.5006
  - Mistral Large 3 vs Amazon Nova Pro: 0.1147
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.5665
  - Claude 3.5 Haiku vs Mistral Large 3: 0.6591
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.7215
  - Llama 3.3 70B vs Mistral Large 3: 0.4344
  - Llama 3.3 70B vs Amazon Nova Pro: 0.4915
  - Mistral Large 3 vs Amazon Nova Pro: 0.9206
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: -0.0194
  - Claude 3.5 Haiku vs Mistral Large 3: 0.2347
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.0949
  - Llama 3.3 70B vs Mistral Large 3: 0.5283
  - Llama 3.3 70B vs Amazon Nova Pro: 0.2256
  - Mistral Large 3 vs Amazon Nova Pro: 0.2664

### Neutral (ctrl)

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.7164 | 0.6184 | 150 |
| Safety | 0.9370 | 0.8575 | 150 |
| Drift | 0.7358 | 0.6108 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.6155
  - Claude 3.5 Haiku vs Mistral Large 3: 0.6500
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.6474
  - Llama 3.3 70B vs Mistral Large 3: 0.5741
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6979
  - Mistral Large 3 vs Amazon Nova Pro: 0.5255
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.8719
  - Claude 3.5 Haiku vs Mistral Large 3: 0.8451
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.8716
  - Llama 3.3 70B vs Mistral Large 3: 0.8190
  - Llama 3.3 70B vs Amazon Nova Pro: 0.8490
  - Mistral Large 3 vs Amazon Nova Pro: 0.8886
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.5912
  - Claude 3.5 Haiku vs Mistral Large 3: 0.7208
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.5294
  - Llama 3.3 70B vs Mistral Large 3: 0.6655
  - Llama 3.3 70B vs Amazon Nova Pro: 0.5928
  - Mistral Large 3 vs Amazon Nova Pro: 0.5651

### Political

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.2986 | 0.3374 | 150 |
| Safety | 0.2135 | 0.2895 | 150 |
| Drift | 0.1000 | 0.3959 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.3989
  - Claude 3.5 Haiku vs Mistral Large 3: 0.1668
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.1565
  - Llama 3.3 70B vs Mistral Large 3: 0.3594
  - Llama 3.3 70B vs Amazon Nova Pro: 0.4638
  - Mistral Large 3 vs Amazon Nova Pro: 0.4792
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.6357
  - Claude 3.5 Haiku vs Mistral Large 3: 0.2523
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.1611
  - Llama 3.3 70B vs Mistral Large 3: 0.2543
  - Llama 3.3 70B vs Amazon Nova Pro: 0.1699
  - Mistral Large 3 vs Amazon Nova Pro: 0.2637
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.1382
  - Claude 3.5 Haiku vs Mistral Large 3: 0.1024
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.1643
  - Llama 3.3 70B vs Mistral Large 3: 0.7685
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6884
  - Mistral Large 3 vs Amazon Nova Pro: 0.5138

### Reformed

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.5848 | 0.5000 | 150 |
| Safety | 0.5479 | 0.5007 | 150 |
| Drift | 0.5983 | 0.4976 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.4233
  - Claude 3.5 Haiku vs Mistral Large 3: 0.3695
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.3305
  - Llama 3.3 70B vs Mistral Large 3: 0.6209
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6442
  - Mistral Large 3 vs Amazon Nova Pro: 0.6117
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.7273
  - Claude 3.5 Haiku vs Mistral Large 3: 0.4107
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.3249
  - Llama 3.3 70B vs Mistral Large 3: 0.4836
  - Llama 3.3 70B vs Amazon Nova Pro: 0.4339
  - Mistral Large 3 vs Amazon Nova Pro: 0.6236
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.3788
  - Claude 3.5 Haiku vs Mistral Large 3: 0.3286
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.4210
  - Llama 3.3 70B vs Mistral Large 3: 0.6501
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6087
  - Mistral Large 3 vs Amazon Nova Pro: 0.5983

### Secure (ctrl)

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.6328 | 0.5340 | 150 |
| Safety | 0.7663 | 0.6753 | 150 |
| Drift | 0.7933 | 0.6698 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.5394
  - Claude 3.5 Haiku vs Mistral Large 3: 0.5777
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.4914
  - Llama 3.3 70B vs Mistral Large 3: 0.5082
  - Llama 3.3 70B vs Amazon Nova Pro: 0.5981
  - Mistral Large 3 vs Amazon Nova Pro: 0.4892
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.7007
  - Claude 3.5 Haiku vs Mistral Large 3: 0.6325
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.5988
  - Llama 3.3 70B vs Mistral Large 3: 0.6963
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6613
  - Mistral Large 3 vs Amazon Nova Pro: 0.7619
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.6441
  - Claude 3.5 Haiku vs Mistral Large 3: 0.6368
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.5906
  - Llama 3.3 70B vs Mistral Large 3: 0.7430
  - Llama 3.3 70B vs Amazon Nova Pro: 0.7177
  - Mistral Large 3 vs Amazon Nova Pro: 0.6866

### Valence (ctrl)

| Dimension | Krippendorff Alpha | Mean Kappa | N Items |
|-----------|-------------------|------------|---------|
| Persona | 0.5122 | 0.4106 | 150 |
| Safety | 0.2565 | 0.2731 | 150 |
| Drift | 0.5420 | 0.4401 | 150 |

**Pairwise Cohen's Kappa:**

- **Persona**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.5270
  - Claude 3.5 Haiku vs Mistral Large 3: 0.2640
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.3050
  - Llama 3.3 70B vs Mistral Large 3: 0.3091
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6285
  - Mistral Large 3 vs Amazon Nova Pro: 0.4301
- **Safety**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.4145
  - Claude 3.5 Haiku vs Mistral Large 3: 0.1938
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.1773
  - Llama 3.3 70B vs Mistral Large 3: 0.1887
  - Llama 3.3 70B vs Amazon Nova Pro: 0.2244
  - Mistral Large 3 vs Amazon Nova Pro: 0.4399
- **Drift**:
  - Claude 3.5 Haiku vs Llama 3.3 70B: 0.4707
  - Claude 3.5 Haiku vs Mistral Large 3: 0.2649
  - Claude 3.5 Haiku vs Amazon Nova Pro: 0.3722
  - Llama 3.3 70B vs Mistral Large 3: 0.3894
  - Llama 3.3 70B vs Amazon Nova Pro: 0.6610
  - Mistral Large 3 vs Amazon Nova Pro: 0.4824

### Overall IRR Summary

| Dimension | Mean Alpha | Mean Kappa | Interpretation |
|-----------|-----------|------------|----------------|
| Persona | 0.4706 | 0.3888 | Fair agreement |
| Safety | 0.6546 | 0.6007 | Substantial agreement |
| Drift | 0.4707 | 0.4445 | Moderate agreement |

## Methodology

- Each judge scored every probe on a 0-3 scale across three dimensions
- Temperature set to 0 for deterministic scoring
- Responses truncated to 2000 characters for consistent context
- Cohen's kappa uses linear weights for ordinal data
- Krippendorff's alpha uses squared-difference metric for ordinal data
- Failed API calls retried with exponential backoff (up to 5 attempts)
