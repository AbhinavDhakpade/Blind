"""
app/alerts/manager.py
----------------------
AlertManager — orchestrates haptic and speech alerts with priority,
cooldown/debouncing, and message templating.

Priority
--------
  1 > 2 > 3

Only the highest-priority alert fires per evaluation.
Two cooldown layers prevent flooding:
  1. a short per-action-type floor (cooldown_ms, default 1200 ms), and
  2. a per-subject cooldown, so the same message about the same person/object
     is not repeated (e.g. "Hello abhinav" at most once a minute).
A higher risk level on an obstacle alert bypasses layer 2 so escalation is
never silenced.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.alerts.haptic import HapticAlertInterface
from app.alerts.speech import SpeechAlertInterface
from app.decision.engine import DecisionAction, ActionType
from app.risk.engine import RiskLevel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AlertManager:
    """Route DecisionActions to the correct hardware alert implementation."""

    def __init__(
        self,
        cfg,
        haptic: HapticAlertInterface,
        speech: SpeechAlertInterface,
    ) -> None:
        self._haptic = haptic
        self._speech = speech
        self._patterns: dict = {}
        self._messages: dict = {}
        self._cooldown_ms: float = getattr(cfg, "cooldown_ms", 1200)
        self._last_alert: dict[ActionType, float] = {}
        # Per-subject cooldowns (seconds); all overridable from the alerts config.
        self._known_cd_s: float    = float(getattr(cfg, "known_person_cooldown_s", 60))
        self._unknown_cd_s: float  = float(getattr(cfg, "unknown_person_cooldown_s", 15))
        self._person_cd_s: float   = float(getattr(cfg, "person_ahead_cooldown_s", 10))
        self._obstacle_cd_s: float = float(getattr(cfg, "obstacle_cooldown_s", 0))  # 0 = off: only cooldown_ms applies
        # key -> (monotonic time, risk level value) of the last alert for that subject
        self._last_keyed: dict[tuple, tuple[float, int]] = {}
        self._load_config(cfg)

    # ------------------------------------------------------------------
    def _load_config(self, cfg) -> None:
        patterns_cfg = getattr(cfg, "haptic_patterns", None)
        if patterns_cfg:
            for level_name in ("low", "medium", "high", "critical"):
                pat = getattr(patterns_cfg, level_name, None)
                if pat is not None:
                    if hasattr(pat, "__dict__"):
                        self._patterns[level_name] = {
                            "pulses": getattr(pat, "pulses", 1),
                            "on_ms":  getattr(pat, "on_ms", 200),
                            "off_ms": getattr(pat, "off_ms", 100),
                        }
                    elif isinstance(pat, dict):
                        self._patterns[level_name] = pat

        msgs_cfg = getattr(cfg, "speech_messages", None)
        if msgs_cfg:
            for key in ("approaching", "obstacle", "crowd", "known_person", "unknown_person"):
                val = getattr(msgs_cfg, key, None)
                if val:
                    self._messages[key] = val

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self._haptic.initialize()
        self._speech.initialize()
        logger.info("AlertManager initialised.")

    # ------------------------------------------------------------------
    def handle(self, action: DecisionAction) -> None:
        """Execute the action, respecting cooldown."""
        if action.action_type == ActionType.NONE:
            return

        if not self._check_cooldown(action.action_type):
            return  # still in cooldown, skip

        key, cooldown_s = self._alert_key(action)
        if not self._check_key(key, cooldown_s, action.risk_level):
            return  # same subject alerted too recently

        self._record_alert(action.action_type)
        self._last_keyed[key] = (time.monotonic(), action.risk_level.value)

        if action.action_type == ActionType.HAPTIC_ALERT:
            self._do_haptic(action)
            # Also produce a brief speech for context
            msg = self._format_message("approaching", action)
            self._speech.speak(msg)

        elif action.action_type == ActionType.SPEECH_OBSTACLE:
            msg = self._format_message("obstacle", action)
            self._speech.speak(msg)

        elif action.action_type == ActionType.IDENTIFY_PERSON:
            if action.person_name:
                msg = self._messages.get("known_person", "Hello {name}").format(
                    name=action.person_name
                )
            elif action.message:
                msg = action.message
            else:
                msg = self._messages.get("unknown_person", "Unknown person nearby")
            self._speech.speak(msg)

    # ------------------------------------------------------------------
    def _do_haptic(self, action: DecisionAction) -> None:
        level_name = action.risk_level.name.lower()
        pattern = self._patterns.get(level_name, {"pulses": 1, "on_ms": 200, "off_ms": 0})
        self._haptic.vibrate(pattern)

    # ------------------------------------------------------------------
    def _format_message(self, key: str, action: DecisionAction) -> str:
        template = self._messages.get(key, action.message or key)
        return template.format(
            object=action.object_type,
            direction=action.direction or "ahead",
            name=action.person_name or "someone",
        )

    # ------------------------------------------------------------------
    def _alert_key(self, action: DecisionAction) -> tuple[tuple, float]:
        """Return (subject key, cooldown seconds) for this action."""
        t = action.action_type
        if t == ActionType.SPEECH_OBSTACLE:
            return (t, action.object_type, action.direction), self._obstacle_cd_s
        if t == ActionType.IDENTIFY_PERSON:
            kind = action.identity_kind or ("known" if action.person_name else "person")
            if kind == "known":
                return (t, "known", action.person_name), self._known_cd_s
            if kind == "unknown":
                return (t, "unknown"), self._unknown_cd_s
            return (t, kind), self._person_cd_s
        # Haptic: only the per-type floor applies.
        return (t, action.object_type, action.direction), 0.0

    def _check_key(self, key: tuple, cooldown_s: float, level: RiskLevel) -> bool:
        if cooldown_s <= 0:
            return True
        last = self._last_keyed.get(key)
        if last is None:
            return True
        last_time, last_level = last
        if time.monotonic() - last_time >= cooldown_s:
            return True
        # Escalation on an obstacle is never silenced.
        return key[0] == ActionType.SPEECH_OBSTACLE and level.value > last_level

    def _check_cooldown(self, action_type: ActionType) -> bool:
        last = self._last_alert.get(action_type, 0.0)
        return (time.monotonic() - last) * 1000.0 >= self._cooldown_ms

    def _record_alert(self, action_type: ActionType) -> None:
        self._last_alert[action_type] = time.monotonic()

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        self._haptic.stop()
        self._haptic.cleanup()
        self._speech.cleanup()
        logger.info("AlertManager cleaned up.")
