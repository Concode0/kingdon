"""GP evidence for issue #140, not a merge-ready implementation.

One-pass strings use generator ranks, never global blade mask identities.
Local support derives temporary binary metadata per multivector operation.
"""

from kingdon.algebra import _swap_blades


class _BladeLayout:
    def __init__(self, signature, start_index, digits, basis=()):
        self.signature = tuple(signature)
        self.generators = tuple(digits[start_index:start_index + len(signature)])
        self.metric = {char: value for char, value in zip(self.generators, signature)}
        self.rank = {char: index for index, char in enumerate(self.generators)}
        self.basis = tuple(basis)
        if self.basis:
            self._canonical = {frozenset(blade[1:]): blade for blade in self.basis}
            self._order = {blade: index for index, blade in enumerate(self.basis)}
            if len(self._canonical) != len(self.basis):
                raise ValueError('Custom basis contains duplicate generator sets.')
        else:
            self._canonical = None

    @classmethod
    def from_algebra(cls, algebra):
        return cls(algebra.signature, algebra.start_index, tuple(algebra.pretty_digits), algebra.basis)

    def canonical(self, generators):
        if self._canonical is not None:
            return self._canonical[frozenset(generators)]
        return 'e' + ''.join(sorted(generators, key=self.rank.__getitem__))

    def normalize(self, blade):
        if not isinstance(blade, str) or not blade.startswith('e'):
            raise ValueError(f'Invalid blade: {blade!r}')
        generators = blade[1:]
        if len(set(generators)) != len(generators) or any(g not in self.metric for g in generators):
            raise ValueError(f'Invalid blade: {blade!r}')
        return self.canonical(generators)

    def grade(self, blade):
        return len(blade) - 1

    def order_key(self, blade):
        return self._order[blade] if self._canonical is not None else (len(blade), blade)


class OnePassStringBlades(_BladeLayout):
    """One-pass merge of default oriented strings; custom orientation is explicit."""

    def product_blades(self, a, b):
        if self.basis:
            left = ''.join(sorted(a[1:], key=self.rank.__getitem__))
            right = ''.join(sorted(b[1:], key=self.rank.__getitem__))
            parity = (_orientation_swaps(a[1:], left) + _orientation_swaps(b[1:], right)) & 1
        else:
            left, right, parity = a[1:], b[1:], 0
        i = j = 0
        output = []
        coefficient = 1
        n, m = len(left), len(right)
        rank, metric = self.rank, self.metric
        while i < n and j < m:
            ga, gb = left[i], right[j]
            ra, rb = rank[ga], rank[gb]
            if ra < rb:
                output.append(ga)
                i += 1
            elif ra > rb:
                output.append(gb)
                parity ^= (n - i) & 1
                j += 1
            else:
                parity ^= (n - i - 1) & 1
                coefficient *= metric[ga]
                i += 1
                j += 1
        if i < n:
            output.append(left[i:])
        if j < m:
            output.append(right[j:])
        default = ''.join(output)
        if self.basis:
            canonical = self._canonical[frozenset(default)]
            parity ^= _orientation_swaps(default, canonical[1:]) & 1
        else:
            canonical = 'e' + default
        return canonical, -coefficient if parity else coefficient


class OutputLabelPoolProbe(OnePassStringBlades):
    """Default-basis output-label pool probe, scoped to one GP operation.

    The ordered generator tuple is only a lookup key for a label that the GP
    already needs to emit. It is not a persistent blade identity or support map.
    """

    def product_with_labels(self, a, b, labels):
        if self.basis:
            raise ValueError('This output-label probe is limited to default bases.')
        left, right = a[1:], b[1:]
        i = j = parity = 0
        output = []
        coefficient = 1
        n, m = len(left), len(right)
        rank, metric = self.rank, self.metric
        while i < n and j < m:
            ga, gb = left[i], right[j]
            ra, rb = rank[ga], rank[gb]
            if ra < rb:
                output.append(ga)
                i += 1
            elif ra > rb:
                output.append(gb)
                parity ^= (n - i) & 1
                j += 1
            else:
                parity ^= (n - i - 1) & 1
                coefficient *= metric[ga]
                i += 1
                j += 1
        output.extend(left[i:])
        output.extend(right[j:])
        key = tuple(output)
        canonical = labels.get(key)
        if canonical is None:
            canonical = 'e' + ''.join(output)
            labels[key] = canonical
        return canonical, -coefficient if parity else coefficient


class CustomOrientationBlades(OnePassStringBlades):
    """Use the supplied oriented basis to prepare string-only orientation maps."""

    def __init__(self, signature, start_index, digits, basis=()):
        if not basis:
            raise ValueError('The custom-orientation probe requires a supplied basis.')
        super().__init__(signature, start_index, digits, basis)
        self._default_gp = OnePassStringBlades(signature, start_index, digits)
        self._input_orientation = {}
        self._output_orientation = {}
        for blade in self.basis:
            default = self._default_gp.normalize(blade)
            parity = _orientation_swaps(blade[1:], default[1:]) & 1
            self._input_orientation[blade] = (default, parity)
            self._output_orientation[default] = (blade, parity)

    def product_blades(self, a, b):
        left, left_parity = self._input_orientation[a]
        right, right_parity = self._input_orientation[b]
        default, coefficient = self._default_gp.product_blades(left, right)
        canonical, output_parity = self._output_orientation[default]
        return (canonical, -coefficient if left_parity ^ right_parity ^ output_parity
                else coefficient)


class LocalSupportBlades(OnePassStringBlades):
    """Operation-local binary support; strings remain the only blade identity."""

    def __init__(self, signature, start_index, digits, basis=()):
        super().__init__(signature, start_index, digits, basis)
        self.negative = sum(1 << i for i, v in enumerate(signature) if v < 0)
        self.null = sum(1 << i for i, v in enumerate(signature) if v == 0)

    def prepare(self, blade, value):
        support = 0
        for generator in blade[1:]:
            support |= 1 << self.rank[generator]
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
        if self.basis:
            return self._canonical[frozenset(g for i, g in enumerate(self.generators)
                                                       if support & (1 << i))]
        chars = []
        while support:
            bit = support & -support
            chars.append(self.generators[bit.bit_length() - 1])
            support ^= bit
        return 'e' + ''.join(chars)

    def product_prepared(self, a, b):
        overlap = a[2] & b[2]
        output = self.output(a[2] ^ b[2])
        if overlap & self.null:
            return output, 0
        parity = ((a[3] & b[2]).bit_count() + (overlap & self.negative).bit_count()
                  + a[4] + b[4]) & 1
        if self.basis:
            default = ''.join(g for g in self.generators if g in output[1:])
            parity ^= _orientation_swaps(default, output[1:]) & 1
        return output, -1 if parity else 1

    def product_blades(self, a, b):
        return self.product_prepared(self.prepare(a, 1), self.prepare(b, 1))



def _orientation_swaps(source, target):
    """Inversion count for reorienting unique generators, with no blade mask."""
    positions = {generator: index for index, generator in enumerate(target)}
    order = [positions[generator] for generator in source]
    return sum(left > right for i, left in enumerate(order) for right in order[i + 1:])


class SwapStringBlades(_BladeLayout):
    """Historical comparator using the existing _swap_blades primitive."""

    def product_blades(self, a, b):
        swaps, remaining, eliminated = _swap_blades(a[1:], b[1:])
        coefficient = 1
        for generator in eliminated:
            coefficient *= self.metric[generator]
        output = self.canonical(remaining)
        swaps += _orientation_swaps(remaining, output[1:])
        return output, coefficient if swaps % 2 == 0 else -coefficient


def gp_with_layout(x, y, layout):
    """Run only GP with a string-native backend and order observed result blades."""
    from kingdon.operators import is_zero

    res = {}
    if isinstance(layout, LocalSupportBlades):
        left_terms = (layout.prepare(a, value) for a, value in x.items())
        right_terms = tuple(layout.prepare(b, value) for b, value in y.items())
        for a in left_terms:
            for b in right_terms:
                output, coefficient = layout.product_prepared(a, b)
                if coefficient:
                    term = a[1] * b[1] if coefficient > 0 else -(a[1] * b[1])
                    res[output] = res[output] + term if output in res else term
    else:
        product = layout.product_blades
        right_terms = tuple(y.items())
        for a, va in x.items():
            for b, vb in right_terms:
                output, coefficient = product(a, b)
                if coefficient:
                    term = va * vb if coefficient > 0 else -(va * vb)
                    res[output] = res[output] + term if output in res else term
    keys = tuple(sorted((key for key, value in res.items() if not is_zero(value)), key=layout.order_key))
    return x.algebra.mvtype.fromkeysvalues(x.algebra, keys, [res[key] for key in keys], raw=True)


def gp_with_output_reuse(x, y, layout, stats=None):
    """Run one-pass GP with a fresh output-label pool for this operation only."""
    from kingdon.operators import is_zero

    labels = {}
    result = {}
    right_terms = tuple(y.items())
    for a, va in x.items():
        for b, vb in right_terms:
            output, sign = layout.product_with_labels(a, b, labels)
            if sign:
                value = va * vb if sign > 0 else -(va * vb)
                result[output] = result[output] + value if output in result else value
    if stats is not None:
        pairs = sum(1 for _ in x.items()) * len(right_terms)
        stats.update(pairs=pairs, hits=pairs - len(labels), misses=len(labels),
                     distinct_labels=len(labels))
    keys = tuple(sorted((key for key, value in result.items() if not is_zero(value)),
                        key=layout.order_key))
    return x.algebra.mvtype.fromkeysvalues(x.algebra, keys, [result[key] for key in keys], raw=True)


class LazyNoTableGP:
    """Restricted default-basis large-algebra probe; sparse MVs and GP only.

    It deliberately has no blade/mask dictionaries, complete blade enumeration,
    type layouts, other operators, or eager blade objects. Use ``sparse`` to
    construct a multivector without invoking the full Algebra constructor.
    """

    def __init__(self, d, signature=None, start_index=1, strategy='one_pass_string'):
        from kingdon.multivector import MultiVector

        if d < 0 or d + start_index > 36:
            raise ValueError('Unsupported generator labels for this lazy GP probe.')
        if signature is not None and len(signature) != d:
            raise ValueError('Signature length must equal the algebra dimension.')
        self.d = d
        self.signature = tuple(signature if signature is not None else (1,) * d)
        self.start_index = start_index
        self.large = True
        self.mvtype = MultiVector
        self._type_layouts = {}
        self.pretty_digits = tuple('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        strategy_class = {'one_pass_string': OnePassStringBlades, 'local_support': LocalSupportBlades,
                          'swap_reference': SwapStringBlades}.get(strategy)
        if strategy_class is None:
            raise ValueError(f'Unknown GP strategy: {strategy!r}')
        self.layout = strategy_class(self.signature, start_index, self.pretty_digits)

    def sparse(self, items):
        normalized = {}
        for blade, value in items.items():
            canonical = self.layout.normalize(blade)
            swaps = _orientation_swaps(blade[1:], canonical[1:])
            term = -value if swaps % 2 else value
            normalized[canonical] = normalized.get(canonical, 0) + term
        keys = tuple(sorted(normalized, key=self.layout.order_key))
        return self.mvtype.fromkeysvalues(self, keys, [normalized[key] for key in keys], raw=True)

    def gp(self, x, y):
        return gp_with_layout(x, y, self.layout)
