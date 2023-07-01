"""Module for controlling Dobot Magician."""

from serial.tools import list_ports
from typing import Optional, Tuple

import pydobot


class Dobot:
    """Class representing a Dobot robotic arm.

    Args:
        port: Serial port to connect to the Dobot. If not provided,
        the first available port will be used.
    """

    def __init__(self, port: Optional[str] = None):
        self._port = port

    def __enter__(self):
        """Context manager entry point for opening the connection to the Dobot.

        Returns:
            Dobot object.
        """
        if not self._port:
            available_ports = list_ports.comports()
            self._port = available_ports[0].device
        print(f"serial port: {self._port}")

        self._device = pydobot.Dobot(port=self._port, verbose=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point for closing the connection to the Dobot."""
        self._device.close()

    def get_current_position(self) -> Tuple[float, float, float]:
        """Gets the current position of the arm.

        Returns:
            Current position (x, y, z) of the arm.
        """
        x, y, z, _, _, _, _, _ = self._device.pose()
        return x, y, z

    def move_to(self, x: float, y: float, z: float, r: float) -> None:
        """Moves the arm to the specified position.

        Args:
            x: Target X-coordinate.
            y: Target Y-coordinate.
            z: Target Z-coordinate.
            r: Target rotation angle.
        """
        self._device.move_to(x, y, z, r, wait=True)
