# Hardware Guide – Vision BOB

## Component List

| Component | Recommended Part | Notes |
|-----------|-----------------|-------|
| Raspberry Pi 5 | RPi 5 (4 GB or 8 GB) | 8 GB preferred for headroom |
| Camera | Raspberry Pi Camera Module 3 | IMX708, 12 MP, libcamera |
| Ultrasonic sensor | HC-SR04 | 2 cm – 4 m range |
| Vibration motor | 3 V / 5 V coin/cylinder motor | With NPN driver circuit |
| Speaker | USB Audio Adapter + small speaker | For espeak-ng TTS |
| NPN Transistor | 2N2222 or BC547 | Motor driver |
| Resistors | 1 kΩ, 2 kΩ, 10 kΩ | Voltage divider + base resistor |
| Flyback diode | 1N4001 or 1N4148 | Motor protection |
| Breadboard / PCB | Standard | |
| Jumper wires | M-F and M-M | |
| Power supply | 5 V, ≥ 5 A (27 W USB-C) | Official Pi 5 supply recommended |

---

## GPIO Pin Reference (BCM Numbering)

```
Raspberry Pi 5 GPIO Header (top view, odd pins left)
Pin 1  [3.3V]   [5V]   Pin 2
Pin 3  [GPIO2]  [5V]   Pin 4       ← 5 V for HC-SR04 VCC
Pin 5  [GPIO3]  [GND]  Pin 6       ← GND
...
Pin 12 [GPIO18] [GND]  Pin 14      ← GPIO18 = vibration motor PWM
...
Pin 16 [GPIO23] [3.3V] Pin 17      ← GPIO23 = HC-SR04 TRIG
Pin 18 [GPIO24] [GND]  Pin 20      ← GPIO24 = HC-SR04 ECHO (voltage-divided)
```

### GPIO Assignment

| Signal | BCM GPIO | Physical Pin | Direction |
|--------|----------|-------------|-----------|
| HC-SR04 TRIG | 23 | 16 | Output |
| HC-SR04 ECHO | 24 | 18 | Input (voltage-divided) |
| Vibration motor | 18 | 12 | Output (PWM) |

These defaults are set in `config/config.yaml` and can be changed.

---

## HC-SR04 Wiring Diagram

```
Raspberry Pi         HC-SR04
──────────           ────────────
5V (Pin 2)  ────────  VCC
GND (Pin 6) ────────  GND
GPIO23      ────────  TRIG
                       ECHO ──┬── 1kΩ ──── GPIO24 (Pin 18)
                               └── 2kΩ ──── GND
```

> ⚠️ **CRITICAL**: The ECHO pin outputs 5 V. The voltage divider is REQUIRED.
> Without it, 5 V on GPIO24 will permanently damage the Raspberry Pi's SoC.

### Voltage divider calculation

V_out = V_in × R2 / (R1 + R2)
      = 5 V × 2000 / (1000 + 2000)
      = 3.33 V   ✓ (within 3.3 V GPIO tolerance)

---

## Vibration Motor Circuit

```
GPIO18 (PWM) ──── 10kΩ ──┬── Base (NPN transistor: 2N2222)
                          │
3.3V ──────────────────── │   (base resistor limits base current)
                          │
                     Collector ──── Motor (–)
                     Emitter  ──── GND
                     
5V ──────────────────── Motor (+) ──┬── 1N4001 (cathode) ──── 5V
                                    └── 1N4001 (anode)   ──── Motor (–)
```

The flyback diode protects the transistor (and Pi) from back-EMF spikes
when the motor is switched off.

---

## Power Budget Estimate

| Component | Current (typical) |
|-----------|-------------------|
| Raspberry Pi 5 (under load) | ~2.5 A |
| Camera Module 3 | ~0.25 A |
| HC-SR04 | ~15 mA |
| Vibration motor | ~100–200 mA |
| **Total** | **~3.1 A** |

Use an official Raspberry Pi 5 power supply (5 V / 5 A = 25 W).
Do NOT power the Pi from a laptop USB port.

---

## Thermal Considerations

The Pi 5 SoC will throttle at 85 °C. Under continuous heavy inference:

```bash
vcgencmd measure_temp
```

Recommendations:
- Use the official Raspberry Pi 5 Active Cooler
- Do not operate in enclosed spaces without airflow
- Consider a heatsink on the SoC

---

## Safety Notes

1. Always disconnect power before changing wiring.
2. Double-check all voltage levels before connecting to GPIO.
3. Use current-limiting resistors for all GPIO outputs.
4. The flyback diode is mandatory — omitting it can cause GPIO damage.
5. This system is an **experimental prototype** and must not be the sole
   navigation safety mechanism.
