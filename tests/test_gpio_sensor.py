"""HC-SR04 driver tests using a fake RPi.GPIO that simulates echo pulses."""
import sys
import time
import types
from types import SimpleNamespace as NS

import pytest

from app.sensors.base import SensorError
from app.sensors.gpio_sensor import HC_SR04Sensor


class FakeGPIO(types.ModuleType):
    """Simulates an HC-SR04: after a trigger pulse the echo pin is high for 2d/c."""
    BCM, OUT, IN = "BCM", "OUT", "IN"

    def __init__(self, distance_m=1.0, responsive=True):
        super().__init__("RPi.GPIO")
        self.distance_m = distance_m
        self.responsive = responsive
        self.triggers = []
        self._t_trig = None

    def setmode(self, m): pass
    def setwarnings(self, f): pass
    def setup(self, pin, mode): pass
    def cleanup(self, pins=None): pass

    def output(self, pin, value):
        if pin == 23 and value is False and self._t_trig is None:
            pass
        if pin == 23 and value is True:
            self._t_trig = time.perf_counter()
            self.triggers.append(self._t_trig)

    def input(self, pin):
        if not self.responsive or self._t_trig is None:
            return 0
        rise = self._t_trig + 0.0005
        fall = rise + 2 * self.distance_m / 343.0
        now = time.perf_counter()
        return 1 if rise <= now < fall else 0


def _cfg(**kw):
    base = dict(trigger_pin=23, echo_pin=24, max_distance_m=4.0, timeout_s=0.04,
                sample_count=3, ping_interval_s=0.06, stale_after_s=0.6)
    base.update(kw)
    return NS(**base)


def _sensor(gpio, **kw):
    s = HC_SR04Sensor(_cfg(**kw))
    s._gpio = gpio
    return s


def test_single_measurement_accuracy():
    s = _sensor(FakeGPIO(distance_m=1.0))
    d = s._single_measurement()
    assert d == pytest.approx(1.0, abs=0.15)     # busy-wait jitter on a PC is fine


def test_no_echo_returns_none_quickly():
    s = _sensor(FakeGPIO(responsive=False))
    t0 = time.perf_counter()
    assert s._single_measurement() is None
    assert time.perf_counter() - t0 < 0.1


def test_pings_are_spaced_at_least_60ms():
    g = FakeGPIO(distance_m=0.8)
    s = _sensor(g)
    s._ranging_cycle()
    gaps = [b - a for a, b in zip(g.triggers, g.triggers[1:])]
    assert len(g.triggers) == 3
    assert all(gap >= 0.055 for gap in gaps)


def test_measure_is_non_blocking_and_returns_latest(monkeypatch):
    fake = FakeGPIO(distance_m=1.2)
    mod = types.ModuleType("RPi"); mod.GPIO = fake
    monkeypatch.setitem(sys.modules, "RPi", mod)
    monkeypatch.setitem(sys.modules, "RPi.GPIO", fake)
    s = HC_SR04Sensor(_cfg())
    s.initialize()
    try:
        first = s.measure()                    # nothing yet -> invalid, instant
        assert first.valid is False
        t0 = time.perf_counter()
        for _ in range(1000):
            s.measure()
        assert time.perf_counter() - t0 < 0.05  # never waits on the hardware
        time.sleep(0.5)
        r = s.measure()
        assert r.valid and r.distance_m == pytest.approx(1.2, abs=0.2)
    finally:
        s.cleanup()


def test_stale_reading_is_invalid():
    s = _sensor(FakeGPIO(), stale_after_s=0.2)
    from app.sensors.base import SensorReading
    s._latest = (SensorReading(1.0, True), time.monotonic() - 1.0)
    r = s.measure()
    assert r.valid is False and "stale" in r.error.lower()


def test_out_of_range_marked_invalid():
    s = _sensor(FakeGPIO(distance_m=3.0), max_distance_m=1.0)
    r = s._ranging_cycle()
    assert r.valid is False


def test_missing_gpio_gives_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "RPi", None)       # forces ImportError
    monkeypatch.setitem(sys.modules, "RPi.GPIO", None)
    with pytest.raises(SensorError) as e:
        HC_SR04Sensor(_cfg()).initialize()
    assert "rpi-lgpio" in str(e.value)
