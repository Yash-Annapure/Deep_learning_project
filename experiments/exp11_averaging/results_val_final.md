# exp11: averaging on the `val,final` sites (full scenes, real monthly labels)

Models: `experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree.pt`, `experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree_seed1.pt`. 5 min in all. '1 seed' = the mean of the two models' own scores. False changes: share of pixels whose Dynamic World class stays the same from one month with radar to the next, where the model's class changes; missed changes: the reverse.

| Seeds · views · months | mIoU | false changes | missed changes | water | trees | low vegetation | flooded vegetation | crops | built | bare |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 seed · 1 view · none | 0.411 | 18.5% | 71.0% | 0.591 | 0.510 | 0.271 | 0.220 | 0.400 | 0.667 | 0.219 |
| 2 seeds · 1 view · none | 0.417 | 17.2% | 72.4% | 0.591 | 0.516 | 0.274 | 0.234 | 0.406 | 0.675 | 0.224 |
| 1 seed · 1 view · despike | 0.416 | 10.3% | 82.0% | 0.590 | 0.514 | 0.274 | 0.226 | 0.411 | 0.673 | 0.224 |
| 2 seeds · 1 view · despike | 0.421 | 9.6% | 82.7% | 0.590 | 0.520 | 0.276 | 0.238 | 0.416 | 0.680 | 0.228 |
| 1 seed · 1 view · 3-month mean | 0.419 | 9.5% | 83.7% | 0.587 | 0.516 | 0.276 | 0.236 | 0.419 | 0.674 | 0.225 |
| 2 seeds · 1 view · 3-month mean | 0.425 | 9.0% | 84.2% | 0.588 | 0.521 | 0.278 | 0.250 | 0.424 | 0.681 | 0.231 |
| 1 seed · 8 views · none | 0.414 | 17.5% | 72.1% | 0.591 | 0.513 | 0.273 | 0.223 | 0.404 | 0.673 | 0.221 |
| 2 seeds · 8 views · none | 0.419 | 16.6% | 73.1% | 0.592 | 0.518 | 0.275 | 0.234 | 0.409 | 0.679 | 0.225 |
| 1 seed · 8 views · despike | 0.418 | 9.9% | 82.6% | 0.591 | 0.517 | 0.275 | 0.227 | 0.414 | 0.678 | 0.226 |
| 2 seeds · 8 views · despike | 0.422 | 9.3% | 83.1% | 0.591 | 0.521 | 0.276 | 0.237 | 0.418 | 0.683 | 0.229 |
| 1 seed · 8 views · 3-month mean | 0.421 | 9.2% | 84.2% | 0.587 | 0.519 | 0.277 | 0.237 | 0.421 | 0.679 | 0.227 |
| 2 seeds · 8 views · 3-month mean | 0.426 | 8.7% | 84.5% | 0.588 | 0.523 | 0.279 | 0.248 | 0.425 | 0.684 | 0.232 |

**Rules (exp11 README):**
1. **2 seeds**: mIoU +0.006, largest class drop +0.001: met
2. **8 views**: mIoU +0.003, largest class drop +0.001: not met
3. **3-month mean instead of despike**: mIoU +0.003, false changes -0.7 points, missed changes +1.7 points: met

**The prototype uses: 2 seeds · 1 view · 3-month mean** (mIoU 0.425, false changes 9.0%, missed 84.2%)
