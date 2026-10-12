#!/usr/bin/env python3

"""Schwinger V5 QFT: funkcional izvora, povezane korelacije i celi prelazi."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree


DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)

WINDOW, STARTS = 96, 8
ALPHAS = (.01, .1, 1.)

j = np.arange(8)
MODES = np.sqrt(2 / 8) * np.cos(
    np.pi * np.arange(1, 8)[:, None]
    * (j[None, :] + .5) / 8
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(numbers) != 7
                or numbers != sorted(numbers)
                or len(set(numbers)) != 7
                or not all(1 <= n <= 39 for n in numbers)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "kombinacija 7/39."
                )

            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1) / 5.


def states(rows):
    return gaps(rows) @ MODES.T


def support(history):
    layout = gaps(history)
    tree = cKDTree(layout)

    nearest = tree.query(
        layout, k=2, p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(nearest, .75)),
        1e-12
    )


def supported(rows, tree, radius):
    distance = tree.query(
        gaps(rows), k=1, p=1
    )[0]

    return distance <= radius + 1e-10


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


class SourceFunctional:
    def __init__(self, changes):
        self.scale = np.maximum(
            np.sqrt(np.mean(changes * changes, axis=0)),
            1e-8
        )

        normalized = changes / self.scale

        # Centriranje unutrasnjih fluktuacija.
        # Ne koristi se kao predikcija prosecnih brojeva.
        self.samples = (
            normalized - normalized.mean(axis=0)
        )
        self.log_count = np.log(len(self.samples))

    def value(self, source):
        source = np.atleast_2d(source)

        return (
            logsumexp(
                source @ self.samples.T,
                axis=1
            )
            - self.log_count
        )

    def connected(self, source):
        if len(source) == 0:
            return np.empty((0, 14))

        blocks = []

        for start in range(0, len(source), 256):
            s = source[start:start + 256]

            terms = (
                s[:, None, :]
                * self.samples[None, :, :]
            )

            one = (
                logsumexp(terms, axis=1)
                - self.log_count
            )

            pair = np.column_stack([
                logsumexp(
                    terms[:, :, j] + terms[:, :, j + 1],
                    axis=1
                )
                - self.log_count
                - one[:, j]
                - one[:, j + 1]
                for j in range(6)
            ])

            total = (
                logsumexp(
                    terms.sum(axis=2),
                    axis=1
                )
                - self.log_count
                - one.sum(axis=1)
            )

            blocks.append(
                np.column_stack((one, pair, total))
            )

        return np.vstack(blocks)


def features(values, reference, previous_change, functional):
    delta = (
        np.atleast_2d(values) - reference
    ) / functional.scale

    previous = previous_change / functional.scale
    connected = functional.connected(delta)

    # 7 prelaza + 7 lokalnih W + 6 povezanih parova
    # + 1 zajednicki povezani clan + 7 memorijskih sprega.
    return np.column_stack((
        delta,
        connected,
        delta * previous
    ))


def prepare(history):
    if len(history) < WINDOW + 2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 2} prethodnih redova."
        )

    history = history[-(WINDOW + 2):]
    values = states(history)
    changes = np.diff(values, axis=0)
    functional = SourceFunctional(changes)

    gram = np.zeros((28, 28))
    rhs = np.zeros(28)

    for t in range(2, len(history)):
        previous = changes[t - 2]

        actual = features(
            values[t:t + 1],
            values[t - 1],
            previous,
            functional
        )

        other = features(
            states(replacements(history[t])),
            values[t - 1],
            previous,
            functional
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast / len(contrast)
        )
        rhs += contrast.sum(axis=0) / len(contrast)

    gram /= WINDOW
    rhs /= WINDOW

    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        values[-1],
        changes[-1],
        functional,
        history[2:]
    )


def fit(prepared, alpha):
    gram, rhs, scale = prepared[:3]

    coefficient = np.linalg.solve(
        gram + alpha * np.eye(28),
        rhs
    ) / scale

    return (
        coefficient,
        prepared[3],
        prepared[4],
        prepared[5]
    )


def scores(rows, model):
    coefficient, reference, previous, functional = model

    result = features(
        states(rows),
        reference,
        previous,
        functional
    ) @ coefficient

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor funkcionala izvora."
        )

    return result


def search(history, model):
    seeds = np.unique(history, axis=0)

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[order[:STARTS]]

    tree, radius = support(history)
    iterations = 0

    while True:
        iterations += 1

        options = np.vstack([
            np.vstack((r, replacements(r)))
            for r in current
        ])

        raw = scores(options, model)
        raw[~supported(options, tree, radius)] = -np.inf

        values = raw.reshape(len(current), 225)
        chosen = np.argmax(values, axis=1)

        better = (
            values[np.arange(len(current)), chosen]
            > values[:, 0] + 1e-10
        )

        if not np.any(better):
            best = int(np.argmax(values[:, 0]))

            return (
                tuple(map(int, current[best])),
                float(values[best, 0]),
                iterations
            )

        blocks = options.reshape(
            len(current), 225, 7
        )

        current[better] = blocks[
            np.arange(len(current))[better],
            chosen[better]
        ]


def predict(prepared, alpha):
    return search(
        prepared[6],
        fit(prepared, alpha)
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows) - test_count
    begin = split - validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW + 2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        p = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo, _, _ = predict(p, alpha)
            validation[j].append(
                hits(combo, rows[t])
            )

        if (t - begin + 1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t - begin + 1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    alpha = ALPHAS[int(np.argmax(means))]

    print(
        f"Validacija: alpha={alpha:g}; "
        f"prosek={max(means):.4f}",
        flush=True
    )

    model_scores, control_scores = [], []

    # Alpha ostaje fiksna tokom testa.
    # Svaka predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _, _ = predict(
            prepare(rows[:t]), alpha
        )

        model_scores.append(
            hits(combo, rows[t])
        )
        control_scores.append(
            hits(rows[t - 1], rows[t])
        )

        if (t - split + 1) % 16 == 0:
            print(
                f"Test: {t - split + 1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split + 1}..{len(rows)}")

    summarize(
        "SCHWINGER V5 QUANTUM FIELD THEORY",
        model_scores
    )
    summarize(
        "Kontrola: prethodno izvuceno kolo",
        control_scores
    )

    p = prepare(rows)
    combo, score, iterations = predict(p, alpha)

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )
    _, radius = support(p[6])

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows) + 1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )
    print(
        "Empirijska podrska: "
        f"L1 poluprecnik={radius:.6f}; kvantil=.75"
    )
    print(
        "Proveren lokalni maksimum dozvoljenih zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "Funkcional izvora je strukturna analogija; "
        "skor nije verovatnoca dobitka."
    )


def self_test():
    base = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39]
    ])

    f = SourceFunctional(
        np.diff(states(base), axis=0)
    )
    zero = np.zeros((1, 7))

    assert np.allclose(f.value(zero), 0.)
    assert np.allclose(f.connected(zero), 0.)

    source = np.array([
        .2, -.3, .1, .4, -.2, .1, .3
    ])

    weights = np.exp(
        f.samples @ source
        - logsumexp(f.samples @ source)
    )
    expected = weights @ f.samples

    numeric = np.zeros(7)
    eps = 1e-6

    for j in range(7):
        plus, minus = source.copy(), source.copy()
        plus[j] += eps
        minus[j] -= eps

        numeric[j] = (
            f.value(plus)[0] - f.value(minus)[0]
        ) / (2 * eps)

    assert np.allclose(
        expected, numeric, atol=1e-8
    )

    history = np.tile(base, (30, 1))
    p = prepare(history)

    for alpha in ALPHAS:
        model = fit(p, alpha)
        a = predict(p, alpha)

        assert a == predict(
            prepare(history), alpha
        )

        tree, radius = support(p[6])
        r = replacements(a[0])
        allowed = supported(r, tree, radius)

        assert (
            scores(r[allowed], model).max(
                initial=-np.inf
            )
            <= a[1] + 1e-9
        )

    changed = history.copy()
    changed[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(changed[:-1]), .1)
    )

    print(
        "Self-test OK: funkcional, povezani clanovi, "
        "izvod, hronologija i pretraga."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=0.1; prosek=1.2708
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
SCHWINGER V5 QUANTUM FIELD THEORY: prosek=1.3750; pogodaka 0..7=[14, 23, 17, 9, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 24,x,26,y,30,z,39
Skor=1.5560880051; iteracija=15; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Funkcional izvora je strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=0.01; prosek=1.2083
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
SCHWINGER V5 QUANTUM FIELD THEORY: prosek=1.4219; pogodaka 0..7=[10, 26, 19, 9, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,x,8,y,16,z,35
Skor=0.8985548564; iteracija=18; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Funkcional izvora je strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=0.01; prosek=1.5833
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
SCHWINGER V5 QUANTUM FIELD THEORY: prosek=1.2188; pogodaka 0..7=[12, 32, 14, 6, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 5,x,29,y,34,z,36
Skor=1.5179180657; iteracija=21; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=3.200000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Funkcional izvora je strukturna analogija; skor nije verovatnoca dobitka.
"""



"""
funkcional izvora i povezane korelacije, uz rangiranje celih kombinacija

V5 računa funkcional izvora iz stvarnih strukturnih prelaza i izdvaja povezane korelacije. Te vrednosti služe za skor cele kombinacije; izlaz se ne dobija prosečenjem brojeva. 

V5 quantum field theory je završena i proverena na sva tri CSV fajla. Koristi funkcional izvora i povezane korelacije za rangiranje celih prelaza — strukturnu primenu ideje iz Schwingerovog rada The Theory of Quantized Fields IV.

CSV	        Sledeća kombinacija	    Prosek pogodaka na 64 kola	Kontrola: prethodno izvučeno kolo
Objedinjeni	24,x,26,y,30,z,39	    1,3750	                    1,2188
Loto	    1,x,8,y,16,z,35	        1,4219	                    1,2969
Loto Plus	5,x,29,y,34,z,36	    1,2188	                    1,2969

Provereni su funkcional, povezane korelacije, hronologija, ponovljivost i lokalni maksimum dozvoljenih zamena. Globalni maksimum nije potvrđen.
"""



"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
