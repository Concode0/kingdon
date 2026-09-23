"""Reproducible issue #140 GP evidence, not a merge-ready implementation.

Run ``.venv/bin/python benchmarks/string_native_gp.py`` from the repository
root to regenerate the complete final JSON dataset. The mask control is
benchmark-only machinery; all direct paths sort only observed result blades.
"""

import argparse
import gc
import itertools
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kingdon import Algebra
from kingdon.blade_experiment import (CustomOrientationBlades, LazyNoTableGP,
                                      LocalSupportBlades, OnePassStringBlades,
                                      OutputLabelPoolProbe, _orientation_swaps,
                                      gp_with_layout, gp_with_output_reuse)
from kingdon.operators import is_zero


def median_seconds(func, minimum=0.018, trials=3):
    count = 1
    while True:
        start = time.perf_counter()
        for _ in range(count):
            func()
        if time.perf_counter() - start >= minimum:
            break
        count *= 2
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(count):
            func()
        times.append((time.perf_counter() - start) / count)
    return statistics.median(times)


class MaskControl(LocalSupportBlades):
    """Current mask arithmetic with observed-result sorting and no sign cache."""

    def __init__(self, algebra):
        super().__init__(algebra.signature, algebra.start_index,
                         tuple(algebra.pretty_digits), algebra.basis)
        self.algebra = algebra
        if algebra.basis:
            # Custom mask bits follow the supplied grade-one basis order.
            self.generators = tuple(blade[1] for blade in algebra.basis if len(blade) == 2)
            self.rank = {g: i for i, g in enumerate(self.generators)}
            digit_rank = {g: i for i, g in enumerate(tuple(algebra.pretty_digits)
                                                  [algebra.start_index:algebra.start_index + algebra.d])}
            self.negative = sum(1 << i for i, g in enumerate(self.generators)
                                if algebra.signature[digit_rank[g]] < 0)
            self.null = sum(1 << i for i, g in enumerate(self.generators)
                            if algebra.signature[digit_rank[g]] == 0)

    def prepare(self, blade, value):
        support = self.algebra.blade2mask[blade]
        suffix = support >> 1
        shift = 1
        while shift < len(self.generators):
            suffix ^= suffix >> shift
            shift <<= 1
        if self.basis:
            default = ''.join(g for g in self.generators if g in blade[1:])
            orientation = _orientation_swaps(blade[1:], default) & 1
        else:
            orientation = 0
        return blade, value, support, suffix, orientation

    def output(self, support):
        return self.algebra.mask2blade[support]


def sparse_keys(d, count, offset=0, start_index=1):
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[start_index:start_index + d]
    # Generate only the low grades needed; never enumerate all 2**d blades.
    pool = []
    for grade in range(1, d + 1):
        pool.extend('e' + ''.join(c) for c in itertools.combinations(digits, grade))
        if len(pool) >= 3 * count:
            break
    if len(pool) < count:
        pool.append('e')
    step = max(1, len(pool) // count)
    return tuple(pool[(offset + i * step) % len(pool)] for i in range(count))


def make_case(d, count, signature, strategy):
    start_index = 0 if signature.count(0) == 1 else 1
    if count == 1:
        generators = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[start_index:start_index + d]
        xkeys = ('e' + generators[0] + generators[1],)
        ykeys = ('e' + generators[1] + generators[2],)
    else:
        xkeys = sparse_keys(d, count, start_index=start_index)
        ykeys = sparse_keys(d, count, offset=3, start_index=start_index)
    xvalues = [i + 1 for i in range(count)]
    yvalues = [i + 2 for i in range(count)]
    if strategy == 'mask_control':
        algebra = Algebra(signature=signature, large=True)
        layout = MaskControl(algebra)
        x = algebra.multivector(keys=xkeys, values=xvalues)
        y = algebra.multivector(keys=ykeys, values=yvalues)
    else:
        algebra = LazyNoTableGP(d, signature=signature, start_index=start_index,
                              strategy=strategy)
        layout = algebra.layout
        x = algebra.sparse(dict(zip(xkeys, xvalues)))
        y = algebra.sparse(dict(zip(ykeys, yvalues)))
    return x, y, layout


def construction_child(d, mode):
    def make_object():
        if mode == 'algebra':
            return Algebra(d, large=True)
        object_ = LazyNoTableGP(d)
        if mode == 'lazy_with_maps':
            digits = tuple(object_.pretty_digits)
            object_.mask2blade = {
                mask: 'e' + ''.join(digits[index + object_.start_index]
                                       for index in range(d) if mask & (1 << index))
                for mask in range(1 << d)
            }
            object_.blade2mask = {blade: mask for mask, blade in object_.mask2blade.items()}
        return object_

    times = []
    for _ in range(5):
        gc.collect()
        start = time.perf_counter()
        first = make_object()
        times.append(time.perf_counter() - start)
        del first
    gc.collect()
    untraced = statistics.median(times)
    tracemalloc.start()
    object_ = make_object()
    current, peak = tracemalloc.get_traced_memory()
    print(json.dumps({'d': d, 'mode': mode, 'seconds': untraced,
                      'retained_mib': current / 2**20, 'peak_mib': peak / 2**20,
                      'blades': len(object_.blade2mask) if hasattr(object_, 'blade2mask') else 0}))


def construction():
    rows = []
    for d in (7, 10, 12, 16):
        for mode in ('algebra', 'lazy_no_table', 'lazy_with_maps'):
            command = [sys.executable, str(Path(__file__).resolve()), '--construct-child', str(d), mode]
            rows.append(json.loads(subprocess.check_output(command, text=True)))
    return rows


def normal_gp():
    rows = []
    for case, factory in (('euclidean4', lambda **kw: Algebra(4, **kw)),
                          ('3dpga', lambda **kw: Algebra.fromname('3DPGA', **kw))):
        expected = None
        for strategy in (None, 'one_pass_string', 'local_support'):
            algebra = factory(_experimental_gp=strategy)
            keys = tuple(algebra.blade2mask)
            x = algebra.multivector(keys=keys[1:9], values=list(range(1, 9)))
            y = algebra.multivector(keys=keys[4:12], values=list(range(2, 10)))
            start = time.perf_counter()
            result = x * y
            cold = time.perf_counter() - start
            observed = dict(result.items())
            if expected is None:
                expected = observed
            assert observed == expected
            rows.append({'case': case, 'strategy': strategy or 'kingdon_baseline',
                         'cold_ms': cold * 1e3,
                         'warm_us': median_seconds(lambda: x * y) * 1e6})
    return rows


def direct_gp():
    rows = []
    signatures = ((7, 'euclidean', (1,) * 7),
                  (12, 'euclidean', (1,) * 12),
                  (12, 'mixed', (1,) * 6 + (-1,) * 6),
                  (12, 'null', (0,) + (1,) * 11),
                  (16, 'euclidean', (1,) * 16))
    for d, metric, signature in signatures:
        for count in (1, 4, 16, 64, 128):
            expected = None
            for strategy in ('one_pass_string', 'local_support', 'mask_control'):
                x, y, layout = make_case(d, count, signature, strategy)
                start = time.perf_counter()
                result = gp_with_layout(x, y, layout)
                first = time.perf_counter() - start
                observed = dict(result.items())
                if expected is None:
                    expected = observed
                assert observed == expected, (d, metric, count, strategy)
                warm = median_seconds(lambda: gp_with_layout(x, y, layout), minimum=0.012)
                rows.append({'d': d, 'metric': metric, 'terms': count,
                             'strategy': strategy, 'first_ms': first * 1e3,
                             'warm_ms': warm * 1e3})
    return rows


def historical_swap():
    """Matched-workload comparison with the earlier _swap_blades miss path."""
    rows = []
    for count in (1, 16, 64, 128):
        x, y, layout = make_case(16, count, (1,) * 16, 'swap_reference')
        rows.append({'d': 16, 'terms': count, 'strategy': 'swap_reference',
                     'warm_ms': median_seconds(lambda: gp_with_layout(x, y, layout),
                                               minimum=0.012) * 1e3})
    return rows


def full_scan_baseline():
    """Isolate today's result-order scan on one sparse 16D direct product."""
    x, y, control = make_case(16, 1, (1,) * 16, 'mask_control')
    algebra = x.algebra
    baseline = algebra.gp(x, y)
    assert dict(baseline.items()) == dict(gp_with_layout(x, y, control).items())
    return {'d': 16, 'terms': 1,
            'kingdon_warm_us': median_seconds(lambda: algebra.gp(x, y),
                                              minimum=0.025, trials=5) * 1e6,
            'observed_order_mask_control_us': median_seconds(
                lambda: gp_with_layout(x, y, control), minimum=0.025, trials=5) * 1e6}


def cached_gp(x, y, layout, cache):
    result = {}
    prepared = isinstance(layout, LocalSupportBlades)
    left = [layout.prepare(a, v) for a, v in x.items()] if prepared else list(x.items())
    right = [layout.prepare(b, v) for b, v in y.items()] if prepared else list(y.items())
    for a in left:
        for b in right:
            key = (a[0], b[0])
            try:
                output, sign = cache.pop(key)
                cache[key] = (output, sign)
                cached_gp.hits += 1
            except KeyError:
                output, sign = (layout.product_prepared(a, b) if prepared
                                else layout.product_blades(a[0], b[0]))
                cache[key] = (output, sign)
                cached_gp.misses += 1
                if len(cache) > 4096:
                    cache.popitem(last=False)
                    cached_gp.evictions += 1
            if sign:
                value = a[1] * b[1] if sign > 0 else -(a[1] * b[1])
                result[output] = result[output] + value if output in result else value
    keys = tuple(sorted((key for key, value in result.items() if not is_zero(value)),
                        key=layout.order_key))
    return x.algebra.mvtype.fromkeysvalues(x.algebra, keys, [result[key] for key in keys], raw=True)


def cache_pressure():
    rows = []
    for count in (32, 64, 128):
        expected = None
        for strategy in ('one_pass_string', 'local_support', 'mask_control'):
            x, y, layout = make_case(12, count, (1,) * 12, strategy)
            cache = OrderedDict()
            cached_gp.hits = cached_gp.misses = cached_gp.evictions = 0
            observed = dict(cached_gp(x, y, layout, cache).items())
            if expected is None:
                expected = observed
            assert observed == expected
            first = (cached_gp.hits, cached_gp.misses, cached_gp.evictions)
            warm = median_seconds(lambda: cached_gp(x, y, layout, cache), minimum=0.014)
            before = (cached_gp.hits, cached_gp.misses, cached_gp.evictions)
            cached_gp(x, y, layout, cache)
            after = (cached_gp.hits, cached_gp.misses, cached_gp.evictions)
            rows.append({'terms': count, 'strategy': strategy, 'warm_ms': warm * 1e3,
                         'first_hits': first[0], 'first_misses': first[1],
                         'warm_hits': after[0] - before[0],
                         'warm_misses': after[1] - before[1],
                         'warm_evictions': after[2] - before[2], 'entries': len(cache)})
    return rows


def output_label_pool():
    """Same 16D operands, accumulation, and observed-result ordering as the control."""
    rows = []
    pool_setup_us = median_seconds(dict, minimum=0.025, trials=7) * 1e6
    for count in (1, 16, 64, 128):
        x, y, one_pass = make_case(16, count, (1,) * 16, 'one_pass_string')
        local = OutputLabelPoolProbe(x.algebra.signature, x.algebra.start_index,
                                     x.algebra.pretty_digits)
        xc, yc, control = make_case(16, count, (1,) * 16, 'mask_control')
        cases = (('one_pass_string', lambda: gp_with_layout(x, y, one_pass)),
                 ('output_label_pool', lambda: gp_with_output_reuse(x, y, local)),
                 ('mask_control', lambda: gp_with_layout(xc, yc, control)))
        stats = {}
        baseline = gp_with_layout(x, y, one_pass)
        expected = dict(baseline.items())
        pooled = gp_with_output_reuse(x, y, local, stats)
        assert dict(pooled.items()) == expected and pooled.keys() == baseline.keys()
        controlled = gp_with_layout(xc, yc, control)
        assert dict(controlled.items()) == expected and controlled.keys() == baseline.keys()
        samples = {strategy: [] for strategy, _ in cases}
        for round_index in range(3):
            for step in range(3):
                strategy, operation = cases[(round_index + step) % 3]
                samples[strategy].append(median_seconds(operation, minimum=0.025,
                                                        trials=5) * 1e3)
        for strategy, operation in cases:
            assert dict(operation().items()) == expected
            rows.append({'d': 16, 'terms': count, 'strategy': strategy,
                         'warm_ms': statistics.median(samples[strategy]),
                         'round_ms': samples[strategy],
                         'pairs': stats['pairs'], 'distinct_output_labels': stats['distinct_labels'],
                         'label_hits': stats['hits'], 'label_misses': stats['misses'],
                         'pool_setup_us': pool_setup_us if strategy == 'output_label_pool' else None})
    return rows


def label_step_cost():
    """Measure the output-label step after the merge has produced its char list."""
    rows = []
    for count in (64, 128):
        x, y, layout = make_case(16, count, (1,) * 16, 'one_pass_string')
        chars = [list(layout.product_blades(a, b)[0][1:])
                 for a, _ in x.items() for b, _ in y.items()]

        def join_every_time():
            for output in chars:
                'e' + ''.join(output)

        def local_pool():
            labels = {}
            for output in chars:
                key = tuple(output)
                canonical = labels.get(key)
                if canonical is None:
                    labels[key] = 'e' + ''.join(output)

        for strategy, operation in (('join_every_time', join_every_time),
                                    ('local_tuple_pool', local_pool)):
            rows.append({'d': 16, 'terms': count, 'strategy': strategy,
                         'pairs': len(chars),
                         'us_per_pair': median_seconds(operation, minimum=0.04,
                                                       trials=7) * 1e6 / len(chars)})
    return rows


def custom_basis():
    """Custom orientation from supplied basis labels, without binary metadata."""
    factories = (
        ('2DPGA', lambda: Algebra.fromname('2DPGA', large=True)),
        ('3DPGA', lambda: Algebra.fromname('3DPGA', large=True)),
        ('custom_3d', lambda: Algebra(3, basis=['e', 'e1', 'e2', 'e3',
                                             'e12', 'e31', 'e23', 'e321'], large=True)),
    )
    primitive_rows, gp_rows, setup_rows = [], [], []
    for case, factory in factories:
        algebra = factory()
        old = OnePassStringBlades.from_algebra(algebra)
        fast = CustomOrientationBlades.from_algebra(algebra)
        control = MaskControl(algebra)
        blades = tuple(algebra.blade2mask)
        pairs = tuple(itertools.product(blades, repeat=2))
        for a, b in pairs:
            ma, mb = algebra.blade2mask[a], algebra.blade2mask[b]
            expected = algebra.mask2blade[ma ^ mb], algebra.signs[ma, mb]
            assert old.product_blades(a, b) == expected
            assert fast.product_blades(a, b) == expected
            assert control.product_blades(a, b) == expected

        def oracle_batch():
            return [(algebra.mask2blade[algebra.blade2mask[a] ^ algebra.blade2mask[b]],
                     algebra.signs[algebra.blade2mask[a], algebra.blade2mask[b]])
                    for a, b in pairs]

        for strategy, product in (
                ('unprepared_custom', old.product_blades),
                ('prepared_custom_orientation', fast.product_blades),
                ('mask_control', control.product_blades)):
            primitive_rows.append({'case': case, 'strategy': strategy, 'pairs': len(pairs),
                                   'us_per_pair': median_seconds(
                                       lambda: [product(a, b) for a, b in pairs],
                                       minimum=0.03, trials=7) * 1e6 / len(pairs)})
        primitive_rows.append({'case': case, 'strategy': 'warm_mask_sign_oracle',
                               'pairs': len(pairs),
                               'us_per_pair': median_seconds(oracle_batch, minimum=0.03,
                                                             trials=7) * 1e6 / len(pairs)})
        setup_rows.append({'case': case, 'basis_blades': len(blades),
                           'fast_metadata_setup_us': median_seconds(
                               lambda: CustomOrientationBlades.from_algebra(algebra),
                               minimum=0.025, trials=7) * 1e6})

        for density, xkeys, ykeys in (
                ('sparse', blades[1::3], blades[2::3]),
                ('full', blades, blades)):
            x = algebra.multivector(keys=xkeys, values=list(range(1, len(xkeys) + 1)))
            y = algebra.multivector(keys=ykeys, values=list(range(2, len(ykeys) + 2)))
            baseline = algebra.gp(x, y)
            expected = dict(baseline.items())
            operations = (
                ('unprepared_custom', lambda: gp_with_layout(x, y, old)),
                ('prepared_custom_orientation', lambda: gp_with_layout(x, y, fast)),
                ('mask_control', lambda: gp_with_layout(x, y, control)),
                ('kingdon_baseline', lambda: algebra.gp(x, y)),
            )
            for strategy, operation in operations:
                result = operation()
                assert dict(result.items()) == expected and result.keys() == baseline.keys()
                gp_rows.append({'case': case, 'density': density, 'terms_x': len(xkeys),
                                'terms_y': len(ykeys), 'strategy': strategy,
                                'warm_us': median_seconds(operation, minimum=0.03,
                                                          trials=7) * 1e6})
    return {'primitive': primitive_rows, 'sparse_gp': gp_rows, 'metadata_setup': setup_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--construct-child', nargs=2, metavar=('D', 'MODE'),
                        help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'benchmarks/string_native_gp_results.json')
    args = parser.parse_args()
    if args.construct_child:
        construction_child(int(args.construct_child[0]), args.construct_child[1])
        return
    data = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor(),
        'implementation': platform.python_implementation(),
        'construction': construction(),
        'normal_gp': normal_gp(),
        'direct_gp': direct_gp(),
        'swap_reference': historical_swap(),
        'full_scan_baseline': full_scan_baseline(),
        'pair_cache': cache_pressure(),
        'output_label_pool': output_label_pool(),
        'label_step_cost': label_step_cost(),
        'custom_basis': custom_basis(),
    }
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
