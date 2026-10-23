import asyncio
import queue
from pathlib import Path

import pytest
from vln_web.web_server import (
    DEFAULT_INSTRUCTION,
    MANUAL_LIMIT_NAMES,
    MPC_CONFIG_NAMES,
    WebServer,
    normalize_server_url,
    parse_manual_limits,
    parse_mpc_config,
)


class Logger:
    def info(self, _message: str) -> None:
        pass


def test_vln_defaults_have_instruction_and_no_server_url():
    web_dir = Path(__file__).parents[1] / "web"
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=queue.Queue(),
        logger=Logger(),
    )

    assert server._snapshot()["vln"]["instruction"] == DEFAULT_INSTRUCTION
    assert server._snapshot()["vln"]["server_url"] == ""
    assert normalize_server_url("") == ""


def test_vln_mode_is_in_snapshot_and_reset():
    web_dir = Path(__file__).parents[1] / "web"
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=queue.Queue(),
        logger=Logger(),
    )
    assert server.vln_mode() == "track"

    server.reset_vln("go to the chair", "objnav")

    assert server.vln_mode() == "objnav"
    assert server._snapshot()["vln"]["mode"] == "objnav"


def test_server_url_is_normalized_and_added_to_snapshot():
    web_dir = Path(__file__).parents[1] / "web"
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=queue.Queue(),
        logger=Logger(),
    )

    server.update_server_url("10.8.204.70:8050")

    assert server._snapshot()["vln"]["server_url"] == (
        "ws://10.8.204.70:8050"
    )


def test_server_url_browser_command_is_normalized_and_queued():
    web_dir = Path(__file__).parents[1] / "web"
    commands = queue.Queue()
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=commands,
        logger=Logger(),
    )

    asyncio.run(
        server._handle_browser_message(
            object(),
            "browser-1",
            {"type": "set_server_url", "server_url": "10.8.204.70:8050"},
        )
    )

    assert commands.get_nowait() == (
        "set_server_url",
        "ws://10.8.204.70:8050",
        "browser-1",
    )


def test_mpc_config_browser_command_is_validated_and_queued():
    web_dir = Path(__file__).parents[1] / "web"
    commands = queue.Queue()
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=commands,
        logger=Logger(),
    )
    config = {name: 1.0 for name in MPC_CONFIG_NAMES}

    asyncio.run(
        server._handle_browser_message(
            object(),
            "browser-1",
            {"type": "set_mpc_config", "config": config},
        )
    )

    assert commands.get_nowait() == (
        "set_mpc_config",
        config,
        "browser-1",
    )
    config["q_yaw"] = 0.0
    assert parse_mpc_config(config)["q_yaw"] == 0.0
    config["w_max"] = 0.0
    with pytest.raises(ValueError, match="w_max"):
        parse_mpc_config(config)


def test_manual_limits_browser_command_is_validated_and_queued():
    web_dir = Path(__file__).parents[1] / "web"
    commands = queue.Queue()
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=commands,
        logger=Logger(),
    )
    config = {name: 1.25 for name in MANUAL_LIMIT_NAMES}

    asyncio.run(
        server._handle_browser_message(
            object(),
            "browser-1",
            {"type": "set_manual_limits", "config": config},
        )
    )

    assert commands.get_nowait() == (
        "set_manual_limits",
        config,
        "browser-1",
    )
    config["linear_accel"] = 0.0
    with pytest.raises(ValueError, match="linear_accel"):
        parse_manual_limits(config)


def test_policy_toggle_does_not_require_manual_control():
    web_dir = Path(__file__).parents[1] / "web"
    commands = queue.Queue()
    server = WebServer(
        host="127.0.0.1",
        port=8088,
        web_dir=web_dir,
        image_topic="camera/color/image_raw",
        manual_linear_limit=1.0,
        manual_angular_limit=1.0,
        manual_linear_accel=1.0,
        manual_angular_accel=1.0,
        commands=commands,
        logger=Logger(),
    )

    asyncio.run(
        server._handle_browser_message(
            object(),
            "browser-1",
            {"type": "robot_action", "action": "toggle_policy"},
        )
    )

    assert commands.get_nowait() == ("mode", "toggle_policy", "browser-1")


@pytest.mark.parametrize(
    "value",
    ["ftp://host", "ws://", "ws://host:70000"],
)
def test_rejects_invalid_server_url(value):
    with pytest.raises(ValueError, match="VLN server URL"):
        normalize_server_url(value)
