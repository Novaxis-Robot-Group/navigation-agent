from vln_web.wifi import parse_wifi_scan


def test_scan_parses_escaped_ssids_and_keeps_strongest_access_point():
    networks = parse_wifi_scan(
        "*:lr-iot:84:WPA2\n"
        ":office\\:5g:62:WPA2\n"
        ":office\\:5g:77:WPA2\n"
        ":guest:40:\n"
    )
    assert networks == [
        {"ssid": "lr-iot", "signal": 84, "security": "WPA2", "active": True},
        {
            "ssid": "office:5g",
            "signal": 77,
            "security": "WPA2",
            "active": False,
        },
        {"ssid": "guest", "signal": 40, "security": "OPEN", "active": False},
    ]
