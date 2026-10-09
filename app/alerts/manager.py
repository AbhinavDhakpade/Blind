"""
app/alerts/manager.py
----------------------
AlertManager — orchestrates haptic and speech alerts with priority,
cooldown/debouncing, and message templating.

Priority
--------
  1 > 2 > 3

Only the highest-priority alert fires per evaluation.
A per-priority cooldown prevents flooding.
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

        self._record_alert(action.action_type)

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
