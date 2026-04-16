# Vendored pybeads

This is a vendored copy of `pybeads` used by this project.

## Local patch

Compatibility fix for NumPy 2.x positional-only arguments:
- `np.reshape(a=..., newshape=...)` -> `np.reshape(..., (...))`
- `np.convolve(a=..., v=...)` -> `np.convolve(..., ...)`
