"""
Minimal DBC parser.

A DBC file is the dictionary for a CAN bus: which messages exist, how often
they're sent, and how to turn the raw bytes into physical values. We only
need two line types:

    BO_ 544 ESP12: 8 ESC
        message id 544, name ESP12, 8 bytes long, sent by node ESC

     SG_ LAT_ACCEL : 0|11@1+ (0.01,-10.23) [-10.23|10.24] "m/s^2"  _4WD,ECS
        signal LAT_ACCEL inside the message above.
        start bit 0, 11 bits long, little-endian (@1), unsigned (+)
        physical = raw * 0.01 + (-10.23)
        valid range -10.23 to 10.24, unit m/s^2

We deliberately do NOT use cantools. The subset we need is small, and a
parser you can read end-to-end is worth more here than a dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Signal:
    name: str
    start_bit: int
    length: int
    factor: float
    offset: float
    minimum: float
    maximum: float
    unit: str

    @property
    def is_flag(self) -> bool:
        """One-bit signals are status flags, not measurements."""
        return self.length == 1


@dataclass
class Message:
    id: int
    name: str
    length: int
    sender: str
    signals: dict[str, Signal] = field(default_factory=dict)


# BO_ <id> <name>: <length> <sender>
_BO = re.compile(r"^BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)")

# SG_ <name> : <start>|<len>@<endian><sign> (<factor>,<offset>) [<min>|<max>] "<unit>"
_SG = re.compile(
    r"^\s*SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@[01]([+-])\s*"
    r"\(([-\d.eE+]+),([-\d.eE+]+)\)\s*"
    r"\[([-\d.eE+]+)\|([-\d.eE+]+)\]\s*"
    r'"([^"]*)"'
)


def parse_dbc(path: str | Path) -> dict[str, Message]:
    """Return {message_name: Message} for every BO_ block in the file.

    Signals attach to whichever BO_ line most recently preceded them,
    which is how the format defines ownership.
    """
    messages: dict[str, Message] = {}
    current: Message | None = None

    for line in Path(path).read_text().splitlines():
        m = _BO.match(line)
        if m:
            msg_id, name, length, sender = m.groups()
            current = Message(int(msg_id), name, int(length), sender)
            messages[name] = current
            continue

        s = _SG.match(line)
        if s and current is not None:
            name, start, length, _sign, factor, offset, lo, hi, unit = s.groups()
            current.signals[name] = Signal(
                name=name,
                start_bit=int(start),
                length=int(length),
                factor=float(factor),
                offset=float(offset),
                minimum=float(lo),
                maximum=float(hi),
                unit=unit,
            )

    return messages


def measurement_signals(messages: dict[str, Message]) -> list[tuple[str, Signal]]:
    """Every (message_name, signal) pair that is a real physical measurement.

    Flags and signals with a degenerate range (min == max) can't drift,
    spike or go out of range in any meaningful way, so the fault injector
    skips them.
    """
    out = []
    for msg in messages.values():
        for sig in msg.signals.values():
            if not sig.is_flag and sig.maximum > sig.minimum and sig.unit:
                out.append((msg.name, sig))
    return out
