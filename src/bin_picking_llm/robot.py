"""Module for controlling Dobot Magician."""

from serial.tools import list_ports
from typing import Optional, Tuple

import pydobot


class Dobot:
    def __init__(self, port: Optional[str] = None):
        self._port = port

    def __enter__(self):
        if not self._port:
            available_ports = list_ports.comports()
            self._port = available_ports[0].device
        print(f"serial port: {self._port}")

        self._device = pydobot.Dobot(port=self._port, verbose=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._device.close()

    def get_current_position(self) -> Tuple[float, float, float]:
        x, y, z, _, _, _, _, _ = self._device.pose()
        return x, y, z

    def move_to(self, x: float, y: float, z: float, r: float) -> None:
        self._device.move_to(x, y, z, r, wait=True)
