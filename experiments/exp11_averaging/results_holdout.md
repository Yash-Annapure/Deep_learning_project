# exp11: averaging on the `holdout` sites (full scenes, real monthly labels)

Models: `experiments/exp10_final/weights/hybrid_r50_ctx_agree_final.pt`, `experiments/exp10_final/weights/hybrid_r50_ctx_agree_final_seed1.pt`. 1 min in all. '1 seed' = the mean of the two models' own scores. False changes: share of pixels whose Dynamic World class stays the same from one month with radar to the next, where the model's class changes; missed changes: the reverse.

| Seeds · views · months | mIoU | false changes | missed changes | water | trees | low vegetation | flooded vegetation | crops | built | bare |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 seed · 1 view · none | 0.481 | 8.8% | 80.7% | 0.806 | 0.632 | 0.417 | 0.060 | 0.406 | 0.803 | 0.242 |
| 2 seeds · 1 view · none | 0.486 | 7.9% | 82.1% | 0.813 | 0.637 | 0.421 | 0.064 | 0.414 | 0.807 | 0.245 |
| 1 seed · 1 view · despike | 0.483 | 4.1% | 91.4% | 0.809 | 0.632 | 0.421 | 0.063 | 0.412 | 0.805 | 0.241 |
| 2 seeds · 1 view · despike | 0.487 | 3.7% | 92.1% | 0.814 | 0.636 | 0.425 | 0.065 | 0.420 | 0.808 | 0.243 |
| 1 seed · 1 view · 3-month mean | 0.485 | 4.0% | 91.8% | 0.809 | 0.632 | 0.424 | 0.070 | 0.415 | 0.805 | 0.242 |
| 2 seeds · 1 view · 3-month mean | 0.489 | 3.6% | 92.2% | 0.814 | 0.636 | 0.428 | 0.073 | 0.422 | 0.808 | 0.244 |
| 1 seed · 8 views · none | 0.485 | 8.0% | 81.9% | 0.812 | 0.635 | 0.422 | 0.061 | 0.411 | 0.806 | 0.244 |
| 2 seeds · 8 views · none | 0.488 | 7.4% | 82.8% | 0.815 | 0.638 | 0.426 | 0.064 | 0.417 | 0.808 | 0.247 |
| 1 seed · 8 views · despike | 0.486 | 3.7% | 91.8% | 0.813 | 0.634 | 0.426 | 0.063 | 0.417 | 0.807 | 0.242 |
| 2 seeds · 8 views · despike | 0.489 | 3.5% | 92.4% | 0.816 | 0.637 | 0.429 | 0.065 | 0.422 | 0.809 | 0.243 |
| 1 seed · 8 views · 3-month mean | 0.488 | 3.7% | 92.1% | 0.813 | 0.634 | 0.429 | 0.070 | 0.419 | 0.807 | 0.242 |
| 2 seeds · 8 views · 3-month mean | 0.491 | 3.4% | 92.5% | 0.815 | 0.637 | 0.432 | 0.073 | 0.424 | 0.809 | 0.244 |
