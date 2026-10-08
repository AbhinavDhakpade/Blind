"""
tests/test_alert_manager.py
----------------------------
Test the AlertManager cooldown and routing logic using mock backends.
"""

import time
import pytest

from app.alerts.haptic import MockHapticAlert
from app.alerts.speech import MockSpeechAlert
from app.alerts.manager import AlertManager
from app.decision.engine import DecisionAction, ActionType
from app.risk.engine import RiskLevel


def _make_cfg(cooldown_ms=500):
    class AlertsCfg:
        cooldown_ms = 500
        haptic_patterns = None
        speech_messages = None
    c = AlertsCfg()
    c.cooldown_ms = cooldown_ms
    return c


def _make_manager(cooldown_ms=50):
    haptic = MockHapticAlert()
    speech = MockSpeechAlert()
    mgr = AlertManager(_make_cfg(cooldown_ms), haptic, speech)
    mgr.initialize()
    return mgr, haptic, speech


def test_haptic_fired_on_p1():
    mgr, haptic, speech = _make_manager()
    action = DecisionAction(
        action_type=ActionType.HAPTIC_ALERT,
        risk_level=RiskLevel.HIGH,
        object_type="car",
        message="car approaching",
    )
    mgr.handle(action)
    assert haptic.last_pattern  # vibrate was called


def test_speech_fired_on_p2():
    mgr, haptic, speech = _make_manager()
    action = DecisionAction(
        action_type=ActionType.SPEECH_OBSTACLE,
        risk_level=RiskLevel.MEDIUM,
        object_type="chair",
        message="Obstacle ahead, chair",
    )
    mgr.handle(action)
    assert speech.last_message != ""


def test_cooldown_prevents_repeat():
    mgr, haptic, speech = _make_manager(cooldown_ms=500)
    action = DecisionAction(
        action_type=ActionType.SPEECH_OBSTACLE,
        risk_level=RiskLevel.MEDIUM,
        object_type="chair",
        message="Obstacle ahead",
    )
    mgr.handle(action)
    first_msg = speech.last_message
    speech.last_message = ""     # reset

    # Immediately re-handle — should be blocked by cooldown
    mgr.handle(action)
    assert speech.last_message == ""  # not fired again


def test_cooldown_expires():
    mgr, haptic, speech = _make_manager(cooldown_ms=50)
    action = DecisionAction(
        action_type=ActionType.SPEECH_OBSTACLE,
        risk_level=RiskLevel.MEDIUM,
        object_type="chair",
        message="Obstacle ahead",
    )
    mgr.handle(action)
    speech.last_message = ""

    time.sleep(0.1)  # wait for cooldown to expire
    mgr.handle(action)
    assert speech.last_message != ""


def test_none_action_no_alert():
    mgr, haptic, speech = _make_manager()
    action = DecisionAction(action_type=ActionType.NONE)
    mgr.handle(action)
    assert speech.last_message == ""
    assert haptic.last_pattern == {}
