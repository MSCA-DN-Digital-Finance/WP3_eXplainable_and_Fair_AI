import pytest
from tsg.generators import (
    LinearTrendGenerator,
    ConstantGenerator,
    PeriodicTrendGenerator
)
from tsg.modifiers import GaussianNoise
import numpy as np

def test_linear_trend_generator_up():
    gen = LinearTrendGenerator(start_price=100, up=True)
    prices = [gen.generate_price() for _ in range(5)]
    assert prices == [101, 102, 103, 104, 105]

def test_linear_trend_generator_down():
    gen = LinearTrendGenerator(start_price=100, up=False)
    prices = [gen.generate_price() for _ in range(5)]
    assert prices == [99, 98, 97, 96, 95]

def test_constant_generator():
    gen = ConstantGenerator()
    val = gen.generate_price(123.45)
    assert val == 123.45
    for _ in range(3):
        assert gen.generate_price(val) == 123.45

def test_periodic_trend_generator_repeatable():
    gen = PeriodicTrendGenerator(start=10.0, amplitude=1.0, frequency=np.pi / 2)
    values = [gen.generate_price() for _ in range(4)]
    expected = [10.0, 11.0, 10.0, 9.0]  # sin(0), sin(pi/2), sin(pi), sin(3pi/2)
    np.testing.assert_allclose(values, expected, rtol=1e-5)

def test_gaussian_noise_perturbs_base():
    base_gen = LinearTrendGenerator(start_price=100)
    noisy_gen = GaussianNoise(base_gen, mu=0.0, sigma=1.0)

    noisy_prices = [noisy_gen.generate_price(None) for _ in range(5)]
    assert all(isinstance(p, float) for p in noisy_prices)
    diffs = [abs(p - (101 + i)) for i, p in enumerate(noisy_prices)]
    assert any(diff > 0 for diff in diffs)  # At least some noise applied

def test_reset_restores_initial_state():
    gen = LinearTrendGenerator(start_price=50, up=True)
    for _ in range(3): gen.generate_price()
    gen.reset()
    assert gen.generate_price() == 51  # After reset, should be 50 + 1
