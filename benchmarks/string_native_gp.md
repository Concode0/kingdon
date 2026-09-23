# Issue #140: blade-string GP evidence

> **Experimental evidence branch, not a merge-ready implementation.**

## Question

[Issue #140](https://github.com/tBuLi/kingdon/issues/140) proposes semantic blade strings as Kingdon's natural blade identity. In the [maintainer's PR #149 comment](https://github.com/tBuLi/kingdon/pull/149#issuecomment-5796767378), the suggested direction goes further: remove binary blade keys and compute products and signs from strings. The uncertainty is `large=True`: normal Kingdon generates numerical functions, but large algebras perform blade-level GP directly at runtime.

This branch measures whether sparse GP can avoid global binary blade mappings, and what direct execution then costs. Only GP is under experiment. The ordinary path remains the default; the private `Algebra._experimental_gp` switch is evidence-only dispatch, not a proposed production API.

## Current Kingdon behavior

`MultiVector` keys are blade strings. GP looks them up in eager `blade2mask` tables, uses `algebra.signs` and mask XOR, converts results through `mask2blade`, and scans the full basis to order result keys. `_compute_sign` itself converts masks back to blade strings and calls `_swap_blades`. Normal algebras mostly pay blade-computation cost when generating a function. With `large=True`, the same operator logic runs on every product, after the full maps have already been built.

## Experimental approaches

| Path | Computation | Global default-basis blade maps? |
|---|---|---|
| `one_pass_string` | Merge canonical generator strings once, counting crossings and repeated generators, applying the metric, and emitting the oriented result. | No |
| `local_support` | Derive an integer support and suffix parity once per input term, use bitwise arithmetic per pair, then synthesize the output string. Strings remain the semantic keys. | No |
| `mask_control` | Use the existing full blade/mask maps with comparable bitwise arithmetic. This benchmark-only control also sorts only observed results. | Yes |
| Custom orientation | Derive string-keyed default-order labels and parity from a supplied oriented basis, then use one-pass string GP. | No binary map; the supplied basis itself has 2^d labels. |
| `lazy_no_table` | Restricted sparse-construction and GP harness for default bases. It omits unrelated `Algebra` features. | No |

The three direct GP paths use the same accumulation and observed-result ordering, with no pair cache in the primary comparison. `local_support` **does use binary computational metadata** within one operation; it does not assign global integer blade identities. The no-table harness's construction time is an upper bound on potential savings for a complete future `Algebra`, not a prediction for one.

## Correctness

Tests compare every basis-blade pair against the existing mask/sign oracle for Euclidean, negative, and null signatures, 2DPGA, named 3DPGA, and a separately oriented custom basis. They also check higher-grade 16D pairs, sparse MV products, observed result order, and representative generated GP source. In named 3DPGA:

```python
e3 * e1 == e31
e1 * e3 == -e31
```

The supplied orientations `e032` and `e021` are retained. The full test suite passes: **561 passed, 180 skipped, 2 xfailed**.

## Main results

The tables come from [the final JSON](string_native_gp_results.json), generated on macOS arm64 with CPython 3.13.13. Warm GP times are calibrated medians in one process, excluding algebra construction. Direct products include result MV construction and sort only observed result blades. Construction uses fresh subprocesses, five untraced timings per mode, and `tracemalloc` for retained memory. Values are workload measurements, not portable speed guarantees.

### Normal generated GP

Warm numerical execution, 8 terms per operand (µs/product):

| Algebra | Current Kingdon | One-pass strings | Local support |
|---|---:|---:|---:|
| 4D Euclidean | 2.938 | 2.862 | 2.869 |
| Named 3DPGA | 2.832 | 2.827 | 2.840 |

Representative generated GP source is identical. Cold generation times are recorded in the JSON but are too small and variable here to establish a ranking. This observation applies to the measured normal/codegen path; it does not explain direct `large=True` timings.

### `large=True` direct GP, no pair cache

Warm sparse MV × MV, ms/product:

| Algebra | Terms | One-pass strings | Local support | Mask control | String / mask |
|---|---:|---:|---:|---:|---:|
| 7D Euclidean | 64×64 | 2.137 | 1.563 | 0.803 | 2.66× |
| 12D Euclidean | 64×64 | 2.596 | 2.251 | 1.219 | 2.13× |
| 16D Euclidean | 16×16 | 0.145 | 0.149 | 0.097 | 1.50× |
| 16D Euclidean | 64×64 | 2.961 | 2.605 | 1.530 | 1.94× |
| 16D Euclidean | 128×128 | 11.264 | 10.007 | 5.612 | 2.01× |

The JSON also contains 1×1 and 4×4 products and 12D mixed negative/null signatures. The one-pass path is roughly twice as fast as the older `_swap_blades` string reference at 16D 16×16 through 128×128. Local support helps more as term count grows, with its preparation cost included, but it does not erase the gap to the mask control. Across measurements on this branch, the larger 16D direct-product string/mask ratio is roughly **1.5–2.1×**. This is a measured runtime tradeoff, not a codegen effect.

### Construction and retained memory

Untraced construction time / retained memory:

| Dimension | Current `Algebra(large=True)` | Minimal `lazy_no_table` harness | Harness plus equivalent full maps |
|---:|---:|---:|---:|
| 7 | 0.462 ms / 0.064 MiB | 0.014 ms / 0.002 MiB | 0.076 ms / 0.015 MiB |
| 12 | 3.728 ms / 0.690 MiB | 0.015 ms / 0.002 MiB | 2.027 ms / 0.547 MiB |
| 16 | 70.310 ms / 9.597 MiB | 0.016 ms / 0.003 MiB | 40.137 ms / 9.453 MiB |

The equivalent 16D maps alone retain about **9.45 MiB** and take about **40 ms** to construct in the minimal harness. The harness omits full `Algebra` initialization and other operators, so its absolute construction time must not be read as achievable by a complete replacement. Result ordering is a separate cost: in the current benchmark, Kingdon's direct path, which scans the full basis, needed about 546 µs for a warm 16D 1×1 GP, versus 2.18 µs for the optimized mask control sorting observed keys. Other path overhead also differs, so this is not an isolated measurement of the scan alone.

### Custom-oriented bases

The supplied-basis metadata path precomputes each oriented label's default-order string and parity, plus the reverse orientation lookup. It uses no blade masks and preserves the supplied canonical labels. Primitive timings (µs/basis-blade pair):

| Basis | Repeated custom sorting/parity | Supplied orientation metadata | Current warm mask/sign oracle |
|---|---:|---:|---:|
| 2DPGA | 1.564 | 0.384 | 0.066 |
| 3DPGA | 1.803 | 0.451 | 0.070 |
| Separately oriented 3D | 1.553 | 0.371 | 0.065 |

For named 3DPGA, 5×5 sparse GP falls from **49.10 to 15.16 µs**; full 16×16 GP falls from **474.85 to 133.29 µs**. Current warm Kingdon takes 6.28 and 32.05 µs for those workloads. The mask/sign oracle has a warmed sign cache, so its primitive timing is a different control from the cache-free direct GP table. Custom metadata setup took about 7–15 µs for the supplied 8- or 16-blade bases.

## Negative findings

- Parsed tuples did not improve GP in the earlier Python prototype. Its [archived exploratory JSON](string_native_gp_initial_results.json) preserves those measurements; tuple code is no longer a candidate in this branch.
- A bounded 4,096-entry complete-pair cache helps repeated working sets that fit, but a 12D 128×128 product has **zero warm hits and 16,384 misses** in the measured pair order. Cache behavior does not settle the representation question.
- An operation-local pool reused 9,644 output labels in a 16D 128×128 product, yet increased latency from **11.269 to 12.436 ms**. A's direct string join costs about 0.034 µs/label; making and looking up a tuple key costs more. Output pooling is retained only as a measured negative control.
- These results do not justify a persistent per-blade object or an unbounded algebra-level cache.

## Interpretation

The evidence strongly supports semantic string blade identity, avoiding eager default-basis `2**d` conversion maps, sorting only observed sparse results, and representing supplied custom orientations with string-keyed metadata. Default-basis sparse GP in the restricted harness needs no hidden global blade table.

The evidence does **not** show that pure string computation is always faster, require transient support bits, reject transient support bits, or establish performance for other operators. `large=True` remains the main open performance tradeoff: optimized strings are far better than the older swap-based miss path, while bitwise computation remains faster for larger direct products in these Python workloads. Normal generated execution being unaffected does not remove that runtime cost.

## What this branch establishes

String blade identity is technically viable; global binary blade maps are unnecessary for sparse default-basis GP; normal generated execution is effectively unchanged; custom-oriented bases can stay string-based; and large direct GP retains a meaningful string-versus-bitwise performance difference. The final architecture belongs in the subsequent #140 design discussion. This branch is **experimental evidence, not a merge-ready implementation**.

## Reproduction

From the repository root:

```sh
.venv/bin/pytest -q tests/test_blade_experiment.py
.venv/bin/pytest -q
.venv/bin/python benchmarks/string_native_gp.py
```

The benchmark command regenerates all current results in `benchmarks/string_native_gp_results.json`. The archived exploratory JSON is historical and is not regenerated by this command.
