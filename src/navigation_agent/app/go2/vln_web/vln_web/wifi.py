"""NetworkManager Wi-Fi operations used by the web control page."""

from __future__ import annotations

import subprocess
from typing import Any


def _split_terse(line: str) -> list[str]:
    fields = []
    current = []
    escaped = False
    for character in line:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    fields.append("".join(current))
    return fields


def parse_wifi_scan(output: str) -> list[dict[str, Any]]:
    """Parse escaped nmcli terse rows and keep the strongest AP per SSID."""
    networks: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = _split_terse(line)
        if len(fields) != 4:
            continue
        active, ssid, signal_text, security = fields
        if not ssid:
            continue
        try:
            signal = max(0, min(100, int(signal_text)))
        except ValueError:
            continue
        item = {
            "ssid": ssid,
            "signal": signal,
            "security": security or "OPEN",
            "active": active == "*",
        }
        existing = networks.get(ssid)
        if (
            existing is None
            or item["active"]
            or signal > int(existing["signal"])
        ):
            networks[ssid] = item
    return sorted(
        networks.values(),
        key=lambda item: (not bool(item["active"]), -int(item["signal"])),
    )


class WifiManager:
    """Run bounded nmcli operations without exposing passwords in argv."""

    def __init__(self, interface: str = "") -> None:
        self.interface = interface.strip()

    def scan(self) -> dict[str, Any]:
        interface = self.interface or self._detect_interface()
        command = [
            "sudo",
            "-n",
            "nmcli",
            "-t",
            "--escape",
            "yes",
            "-f",
            "IN-USE,SSID,SIGNAL,SECURITY",
            "device",
            "wifi",
            "list",
            "--rescan",
            "yes",
            "ifname",
            interface,
        ]
        output = self._run(command, timeout_s=15.0)
        networks = parse_wifi_scan(output)
        current = next(
            (str(item["ssid"]) for item in networks if item["active"]), ""
        )
        return {
            "available": True,
            "interface": interface,
            "current_ssid": current,
            "networks": networks,
            "error": "",
        }

    def connect(self, ssid: str, password: str) -> dict[str, Any]:
        ssid = ssid.strip()
        if not ssid or len(ssid.encode("utf-8")) > 32:
            raise ValueError("Wi-Fi SSID must contain 1 to 32 bytes")
        if "\n" in password or "\r" in password or len(password) > 128:
            raise ValueError("invalid Wi-Fi password")
        interface = self.interface or self._detect_interface()
        command = [
            "sudo",
            "-n",
            "nmcli",
            "--wait",
            "30",
            "--ask",
            "device",
            "wifi",
            "connect",
            ssid,
            "ifname",
            interface,
        ]
        self._run(command, input_text=f"{password}\n", timeout_s=35.0)
        return self.scan()

    def _detect_interface(self) -> str:
        output = self._run(
            [
                "sudo",
                "-n",
                "nmcli",
                "-t",
                "--escape",
                "yes",
                "-f",
                "DEVICE,TYPE",
                "device",
                "status",
            ],
            timeout_s=5.0,
        )
        for line in output.splitlines():
            fields = _split_terse(line)
            if len(fields) == 2 and fields[1] == "wifi" and fields[0]:
                return fields[0]
        raise RuntimeError("no NetworkManager Wi-Fi interface found")

    @staticmethod
    def _run(
        command: list[str],
        *,
        timeout_s: float,
        input_text: str | None = None,
    ) -> str:
        try:
            result = subprocess.run(
                command,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("nmcli or passwordless sudo is unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Wi-Fi operation timed out") from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or "nmcli operation failed")
        return result.stdout
