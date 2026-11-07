# `hybrid_ctx_agree` with a convolutional CRF (validation, 12 sites, real monthly labels)

| CRF | mIoU | water | trees | low vegetation | flooded vegetation | crops | built | bare |
|---|---|---|---|---|---|---|---|---|
| none (argmax) | 0.455 | 0.547 | 0.332 | 0.394 | 0.397 | 0.357 | 0.791 | 0.364 |
| mild: window 7 px, theta_xy 3.0, theta_i 0.1 | 0.456 | 0.544 | 0.327 | 0.396 | 0.402 | 0.358 | 0.793 | 0.371 |
| strong: window 11 px, theta_xy 5.0, theta_i 0.1 | 0.454 | 0.536 | 0.320 | 0.396 | 0.403 | 0.357 | 0.793 | 0.375 |
