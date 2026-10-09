"""
tests/test_sensor.py
---------------------
Test ultrasonic sensor: valid distance, invalid distance, timeout simulation.
"""

import pytest
from app.sensors.mock_sensor import MockUltrasonicSensor
from app.sensors.base import SensorReading


class _MockCfg:
    backend = "mock"
    trigger_pin = 23
    echo_pin = 24
    max_distance_m = 4.0
    timeout_s = 0.04
    sample_count = 3


def test_mock_sensor_returns_valid_reading():
    sensor = MockUltrasonicSensor(_MockCfg())
    sensor.initialize()
    reading = sensor.measure()
    assert reading.valid is True
    assert reading.distance_m is not None
    assert 0.4 <= reading.distance_m <= 3.6
    sensor.cleanup()


def test_simulated_timeout():
    sensor = MockUltrasonicSensor(_MockCfg())
    sensor._fail_after = 0   # fail immediately
    sensor.initialize()
    reading = sensor.measure()
    assert reading.valid is False
    assert reading.distance_m is None
    sensor.cleanup()


def test_multiple_readings_in_range():
    sensor = MockUltrasonicSensor(_MockCfg())
    sensor.initialize()
    readings = [sensor.measure() for _ in range(10)]
    valid_readings = [r for r in readings if r.valid]
    assert len(valid_readings) > 0
    for r in valid_readings:
        assert 0.0 < r.distance_m < 5.0
    sensor.cleanup()
