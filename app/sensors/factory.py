"""app/sensors/factory.py"""
from __future__ import annotations
from app.sensors.base import UltrasonicSensorInterface


def create_sensor(cfg) -> UltrasonicSensorInterface:
    backend = str(getattr(cfg, "backend", "mock")).lower()

    if backend == "gpio":
        from app.sensors.gpio_sensor import HC_SR04Sensor
        return HC_SR04Sensor(cfg)

    if backend == "mock":
        from app.sensors.mock_sensor import MockUltrasonicSensor
        return MockUltrasonicSensor(cfg)

    raise ValueError(f"Unknown sensor backend: {backend!r}. Valid: 'gpio', 'mock'.")
