"""唯一具体实现装配点；新增设备或模型在这里登记。"""

from collections.abc import Callable
from dataclasses import dataclass

from navigation_agent.contracts.agent_backend import RobotBackend
from navigation_agent.navigation.backend import NavigationBackend
from navigation_agent.navigation.session import RemoteNavigationSession
from navigation_agent.app.mujoco.connection import ConsoleConnection
from navigation_agent.app.deployments.settings import BackendSettings


@dataclass(frozen=True)
class RobotAdapter:
    config: type[BackendSettings]
    create: Callable[..., RobotBackend]


_ROBOTS: dict[str, RobotAdapter] = {}


def register_robot(name: str, adapter: RobotAdapter):
    """入口显式登记配置与构造函数，禁止覆盖已有实现。"""
    if not name or name in _ROBOTS:
        raise ValueError('本体名称为空或重复：' + name)
    if not issubclass(adapter.config, BackendSettings):
        raise TypeError('本体配置必须继承 BackendSettings')
    _ROBOTS[name] = adapter


def get_robot(name: str) -> RobotAdapter:
    if name not in _ROBOTS:
        raise ValueError('未注册的本体：' + name)
    return _ROBOTS[name]


def create_backend(settings: BackendSettings, *, saved_motion=None, base_url=None, model_url=None) -> RobotBackend:
    adapter = get_robot(settings.robot.backend)
    return adapter.create(settings, saved_motion=saved_motion, base_url=base_url, model_url=model_url)


def create_go2(settings, *, saved_motion=None, base_url=None, model_url=None):
    from navigation_agent.app.go2.connection import Go2Connection
    from navigation_agent.robots.go2.robot import Go2Robot, Go2NavigationSession

    robot_settings = settings.robot
    names = ('look_angle_deg', 'look_tolerance_deg', 'look_angular_min',
             'look_angular_max', 'look_timeout_s', 'look_stall_s', 'look_gain', 'look_progress_deg')
    look = saved_motion.look.model_dump() if saved_motion else {
        name: getattr(robot_settings, name + '_rad_s' if name in ('look_angular_min', 'look_angular_max') else name)
        for name in names
    }
    look.pop('navigation_seconds', None)
    protection = saved_motion.protection if saved_motion else settings.protection
    robot = Go2Robot(Go2Connection(base_url or robot_settings.base_url,
                     command_timeout=protection.command_timeout_s), **look, protection=protection)
    backend = NavigationBackend(
        robot, Go2NavigationSession(robot, model_url or settings.vln.server_url),
        max_attempts=settings.vln.max_attempts,
        **settings.navigation.model_dump(),
    )
    backend.blocked_timeout_s = protection.blocked_timeout_s
    backend.progress.settings = protection
    return backend


def create_mujoco(settings, *, saved_motion=None, base_url=None, model_url=None):
    from navigation_agent.robots.mujoco.robot import MujocoRobot
    channel = ConsoleConnection(base_url or settings.robot.base_url)
    return NavigationBackend(
        MujocoRobot(channel), RemoteNavigationSession(channel, model_url or settings.vln.server_url),
        **settings.navigation.model_dump(),
    )


register_robot('go2', RobotAdapter(BackendSettings, create_go2))
register_robot('mujoco', RobotAdapter(BackendSettings, create_mujoco))


def create_vln_client(settings):
    """在运行客户端的进程按配置装配。"""
    from navigation_agent.app.deployments.vln import get_vln, validate_vln
    adapter = get_vln(settings.backend)
    options = validate_vln(settings.backend, settings.server_url, settings.options)
    return adapter.create(settings.server_url, options)


def create_simulation_components(simulation=None, *, camera_height_m=0.198):
    """仿真服务的具体装配；Agent 进程不会加载 MuJoCo。"""
    from vln_mujoco.model import SCENE_DATASET, SCENE_ID, SCENE_NAME
    from vln_mujoco.mpc import MpcTracker
    from navigation_agent.robots.mujoco.executor import MujocoExecutor

    if simulation is None:
        from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot
        from navigation_agent.robots.mujoco.runtime import RealtimeSimulation
        simulation = RealtimeSimulation(SafeTurtleBot(camera_height_m=camera_height_m))
    if getattr(getattr(simulation, 'robot', None), 'envelope', None) is not None:
        from navigation_agent.robots.mujoco.local_navigation import LocalNavigationTracker
        tracker = LocalNavigationTracker(simulation)
    else:
        tracker = MpcTracker()
    return simulation, MujocoExecutor(simulation, tracker), {
        'id': SCENE_ID, 'name': SCENE_NAME, 'dataset': SCENE_DATASET,
    }
