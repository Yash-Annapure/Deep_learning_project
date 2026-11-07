# Scores on the holdout sites (scored once)

| Model | mIoU | mIoU tropics | mIoU new regions | water | trees | low vegetation | flooded vegetation | crops | built | bare |
|---|---|---|---|---|---|---|---|---|---|---|
| `experiments/exp4_50sites/weights/hybrid_ctx_agree.pt` | 0.433 | 0.357 | 0.437 | 0.743 | 0.568 | 0.364 | 0.037 | 0.339 | 0.793 | 0.189 |
| `experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree.pt` | 0.420 | 0.364 | 0.413 | 0.706 | 0.554 | 0.334 | 0.037 | 0.347 | 0.782 | 0.182 |
| `experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree_seed1.pt` | 0.425 | 0.368 | 0.414 | 0.733 | 0.561 | 0.316 | 0.046 | 0.321 | 0.794 | 0.206 |
| `experiments/exp8_ssl4eo/weights/hybrid_r50_ssl4eo_ctx_agree.pt` | 0.414 | 0.333 | 0.422 | 0.690 | 0.525 | 0.366 | 0.021 | 0.326 | 0.797 | 0.169 |
| `experiments/exp8_ssl4eo/weights/hybrid_r50_ssl4eo_ctx_agree_seed1.pt` | 0.414 | 0.342 | 0.424 | 0.700 | 0.539 | 0.373 | 0.042 | 0.298 | 0.801 | 0.142 |
| `experiments/exp10_final/weights/hybrid_r50_ctx_agree_final.pt` | 0.481 | 0.366 | 0.491 | 0.786 | 0.633 | 0.415 | 0.068 | 0.416 | 0.806 | 0.243 |
| `experiments/exp10_final/weights/hybrid_r50_ctx_agree_final_seed1.pt` | 0.481 | 0.366 | 0.495 | 0.794 | 0.633 | 0.428 | 0.072 | 0.402 | 0.808 | 0.233 |
