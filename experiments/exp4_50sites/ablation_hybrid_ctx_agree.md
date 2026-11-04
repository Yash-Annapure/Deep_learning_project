# What `hybrid_ctx_agree` relies on: validation IoU with one part of its input taken away

12 validation sites, real monthly labels only. Each row replaces one part of the input (see the second column); the numbers are the change in IoU against the full input, so **more negative = relied on more**. Baseline IoU in the first row.

| Taken away | What it tests | mIoU | water | trees | low vegetation | flooded vegetation | crops | built | bare | Hit hardest |
|---|---|---|---|---|---|---|---|---|---|---|
| everything (baseline) | the model as trained | 0.455 | 0.547 | 0.332 | 0.394 | 0.397 | 0.357 | 0.791 | 0.364 | – |
| this month's VV → average | how much it needs VV (strongest echo: built, bare, water) | -0.186 | -0.189 | -0.265 | -0.065 | -0.332 | -0.095 | -0.049 | -0.306 | flooded vegetation -0.33, bare -0.31 |
| this month's VH → average | how much it needs VH (volume scattering: vegetation) | -0.247 | -0.262 | -0.216 | -0.178 | -0.367 | -0.116 | -0.230 | -0.359 | flooded vegetation -0.37, bare -0.36 |
| this month's radar → average | whether the month itself matters at all, or the yearly context does the work | -0.238 | -0.286 | -0.331 | -0.076 | -0.393 | -0.155 | -0.090 | -0.335 | flooded vegetation -0.39, bare -0.34 |
| this month → the yearly median | whether it reads THIS month, or mostly the typical year (the month's own changes, e.g. a flood, are gone) | -0.132 | -0.090 | -0.080 | -0.035 | -0.160 | -0.336 | -0.031 | -0.193 | crops -0.34, bare -0.19 |
| yearly median → average | how much it needs the yearly context | -0.374 | -0.381 | -0.070 | -0.394 | -0.357 | -0.265 | -0.791 | -0.363 | built -0.79, low vegetation -0.39 |
| this month blurred (σ = 20 m) | how much it relies on fine texture and sharp edges in the month's radar | -0.117 | -0.070 | -0.169 | -0.012 | -0.292 | -0.081 | -0.048 | -0.144 | flooded vegetation -0.29, trees -0.17 |
