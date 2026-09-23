"""Finite correctness oracle for the issue #140 GP evidence branch."""

import inspect

import pytest

from kingdon import Algebra
from kingdon.blade_experiment import (CustomOrientationBlades, LazyNoTableGP,
                                      OutputLabelPoolProbe, OnePassStringBlades,
                                      LocalSupportBlades, SwapStringBlades,
                                      gp_with_layout, gp_with_output_reuse)


CASES = [
    lambda: Algebra(3),
    lambda: Algebra(1, 2),
    lambda: Algebra(2, 0, 1),
    lambda: Algebra(3, 1, 1),
    lambda: Algebra.fromname('2DPGA'),
    lambda: Algebra.fromname('3DPGA'),
    lambda: Algebra(3, basis=['e', 'e1', 'e2', 'e3', 'e12', 'e31', 'e23', 'e321']),
]


@pytest.mark.parametrize('make_algebra', CASES)
def test_every_blade_pair_against_existing_mask_oracle(make_algebra):
    algebra = make_algebra()
    backends = [cls.from_algebra(algebra) for cls in
                (SwapStringBlades, OnePassStringBlades, LocalSupportBlades)]
    for a, mask_a in algebra.blade2mask.items():
        for b, mask_b in algebra.blade2mask.items():
            expected = (algebra.mask2blade[mask_a ^ mask_b], algebra.signs[mask_a, mask_b])
            for backend in backends:
                assert backend.product_blades(a, b) == expected, (a, b, type(backend))


def test_named_3dpga_orientation():
    algebra = Algebra.fromname('3DPGA')
    for cls in (OnePassStringBlades, LocalSupportBlades):
        backend = cls.from_algebra(algebra)
        assert backend.product_blades('e3', 'e1') == ('e31', 1)
        assert backend.product_blades('e1', 'e3') == ('e31', -1)
        assert backend.normalize('e023') == 'e032'
        assert backend.normalize('e012') == 'e021'


def test_high_grade_16d_blade_pairs():
    algebra = Algebra(16, large=True)
    masks = (0, 1, 3, 0x1234, 0x3333, 0x5555, 0xAAAA, 0xFFFF)
    for cls in (OnePassStringBlades, LocalSupportBlades):
        backend = cls.from_algebra(algebra)
        for ma in masks:
            for mb in masks:
                a, b = algebra.mask2blade[ma], algebra.mask2blade[mb]
                assert backend.product_blades(a, b) == (
                    algebra.mask2blade[ma ^ mb], algebra.signs[ma, mb])


@pytest.mark.parametrize('make_algebra', CASES[:4])
def test_operation_local_output_labels_match_one_pass(make_algebra):
    algebra = make_algebra()
    layout = OutputLabelPoolProbe.from_algebra(algebra)
    labels = {}
    for a, ma in algebra.blade2mask.items():
        for b, mb in algebra.blade2mask.items():
            output, sign = layout.product_with_labels(a, b, labels)
            assert (output, sign) == (algebra.mask2blade[ma ^ mb], algebra.signs[ma, mb])
    assert len(labels) == len(algebra.blade2mask)


def test_output_label_pool_is_operation_local_and_has_hits():
    lazy = LazyNoTableGP(16)
    layout = OutputLabelPoolProbe(lazy.signature, lazy.start_index, lazy.pretty_digits)
    x = lazy.sparse({'e12': 2, 'e13': 3, 'e23': 4})
    y = lazy.sparse({'e12': 5, 'e13': 6, 'e23': 7})
    stats = {}
    pooled = gp_with_output_reuse(x, y, layout, stats)
    baseline = gp_with_layout(x, y, lazy.layout)
    assert dict(pooled.items()) == dict(baseline.items())
    assert pooled.keys() == baseline.keys()
    assert stats['pairs'] == 9
    assert stats['hits'] + stats['misses'] == 9
    assert stats['distinct_labels'] == stats['misses']
    assert stats['hits'] > 0
    again = {}
    gp_with_output_reuse(x, y, layout, again)
    assert again == stats
    assert not hasattr(layout, 'blade2mask')
    assert not hasattr(layout, 'mask2blade')


@pytest.mark.parametrize('make_algebra', CASES[4:])
def test_fast_custom_basis_exhaustively_matches_oracle(make_algebra):
    algebra = make_algebra()
    old = OnePassStringBlades.from_algebra(algebra)
    fast = CustomOrientationBlades.from_algebra(algebra)
    for a, ma in algebra.blade2mask.items():
        for b, mb in algebra.blade2mask.items():
            expected = (algebra.mask2blade[ma ^ mb], algebra.signs[ma, mb])
            assert old.product_blades(a, b) == expected
            assert fast.product_blades(a, b) == expected
    assert len(fast._input_orientation) == len(algebra.basis)
    assert len(fast._output_orientation) == len(algebra.basis)


def test_fast_custom_orientation_and_sparse_gp():
    for make_algebra in CASES[4:]:
        algebra = make_algebra()
        fast = CustomOrientationBlades.from_algebra(algebra)
        keys = tuple(algebra.blade2mask)
        x = algebra.multivector(keys=keys[::2], values=list(range(1, len(keys[::2]) + 1)))
        y = algebra.multivector(keys=keys[1::2], values=list(range(2, len(keys[1::2]) + 2)))
        result = gp_with_layout(x, y, fast)
        baseline = x * y
        assert dict(result.items()) == dict(baseline.items())
        assert result.keys() == baseline.keys()
    named = CustomOrientationBlades.from_algebra(Algebra.fromname('3DPGA'))
    assert named.product_blades('e3', 'e1') == ('e31', 1)
    assert named.product_blades('e1', 'e3') == ('e31', -1)
    assert named.normalize('e023') == 'e032'
    assert named.normalize('e012') == 'e021'


@pytest.mark.parametrize('backend', ['one_pass_string', 'local_support'])
@pytest.mark.parametrize('large', [False, True])
def test_integrated_gp_matches_baseline(backend, large):
    for factory in (lambda **kw: Algebra(2, 1, 1, **kw),
                    lambda **kw: Algebra.fromname('3DPGA', **kw)):
        baseline = factory(large=large)
        candidate = factory(large=large, _experimental_gp=backend)
        keys = tuple(baseline.blade2mask)
        for left, right in ((keys[1:5], keys[4:9]),
                            (keys[::3], keys[2::4]), (keys, keys)):
            x_old = baseline.multivector(keys=left, values=list(range(1, len(left) + 1)))
            y_old = baseline.multivector(keys=right, values=list(range(2, len(right) + 2)))
            x_new = candidate.multivector(keys=left, values=list(range(1, len(left) + 1)))
            y_new = candidate.multivector(keys=right, values=list(range(2, len(right) + 2)))
            assert dict((x_new * y_new).items()) == dict((x_old * y_old).items())


def test_lazy_large_sparse_gp_has_no_blade_tables():
    baseline = Algebra(16, large=True)
    for strategy in ('swap_reference', 'one_pass_string', 'local_support'):
        lazy = LazyNoTableGP(16, strategy=strategy)
        assert not hasattr(lazy, 'blade2mask')
        assert not hasattr(lazy, 'mask2blade')
        x = lazy.sparse({'e1': 2, 'e31': 3})
        y = lazy.sparse({'e2': 5, 'e3': 7})
        old = baseline.multivector(dict(x.items())) * baseline.multivector(dict(y.items()))
        assert dict(lazy.gp(x, y).items()) == dict(old.items())


def test_generated_gp_source_matches_baseline():
    for factory in (lambda **kw: Algebra(3, **kw),
                    lambda **kw: Algebra.fromname('3DPGA', **kw)):
        baseline = factory()
        keys = tuple(baseline.blade2mask)
        xkeys, ykeys = keys[1:7], keys[4:10]
        x_old = baseline.multivector(keys=xkeys, values=[1] * len(xkeys))
        y_old = baseline.multivector(keys=ykeys, values=[1] * len(ykeys))
        baseline.gp(x_old, y_old)
        old_source = inspect.getsource(baseline.gp[x_old, y_old].func)
        for backend in ('one_pass_string', 'local_support'):
            candidate = factory(_experimental_gp=backend)
            x_new = candidate.multivector(keys=xkeys, values=[1] * len(xkeys))
            y_new = candidate.multivector(keys=ykeys, values=[1] * len(ykeys))
            candidate.gp(x_new, y_new)
            assert old_source == inspect.getsource(candidate.gp[x_new, y_new].func)


def test_lazy_large_negative_null_and_high_labels():
    lazy = LazyNoTableGP(3, signature=(0, 1, -1), start_index=0, strategy='local_support')
    assert dict(lazy.gp(lazy.sparse({'e0': 1}), lazy.sparse({'e0': 1})).items()) == {}
    assert dict(lazy.gp(lazy.sparse({'e2': 1}), lazy.sparse({'e2': 1})).items()) == {'e': -1}
    high = LazyNoTableGP(16, strategy='one_pass_string')
    assert dict(high.gp(high.sparse({'eG': 1}), high.sparse({'e1': 1})).items()) == {'e1G': -1}
