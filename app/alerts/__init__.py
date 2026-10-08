"""app/alerts/__init__.py"""
from app.alerts.manager import AlertManager
from app.alerts.haptic import HapticAlertInterface
from app.alerts.speech import SpeechAlertInterface

__all__ = ["AlertManager", "HapticAlertInterface", "SpeechAlertInterface"]
