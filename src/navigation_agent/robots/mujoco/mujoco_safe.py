"""为原版 LightNav MuJoCo 场景添加 TurtleBot 碰撞约束。

沿用固定版本的场景和协议，在项目执行层补充局部避障，不修改上游代码。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from vln_mujoco.model import PHYSICS_DT, compile_model, load_scene_xml
from vln_mujoco.robots.turtlebot import TurtleBotBackend, _robot_body, WHEEL_RADIUS, WHEEL_TRACK
from vln_mujoco.robots.base import RobotState

BODY_RADIUS = 0.11
BODY_BOTTOM = 0.01
BODY_TOP = 0.235
CLEARANCE = 0.005
MAX_TRANSLATION_STEP = 0.0025


class SafeTurtleBot(TurtleBotBackend):
    def __init__(self, source=None, *, camera_height_m=0.198):
        root = load_scene_xml(source)
        body = _robot_body()
        if not math.isfinite(camera_height_m) or not 0.05 <= camera_height_m <= 2.0:
            raise ValueError('camera_height_m 必须在 0.05–2.0 米之间')
        # 只调整虚拟相机离地高度；不改变车身和碰撞包络。
        camera = body.find(".//camera[@name='robot_rgb']")
        if camera is None:
            raise RuntimeError('上游 TurtleBot 缺少 robot_rgb 相机')
        position = camera.get('pos').split()
        position[2] = str(camera_height_m - WHEEL_RADIUS)
        camera.set('pos', ' '.join(position))

        # 保留外观，用直立圆柱包住车轮、相机和支架。
        # 圆柱底部留一厘米间隙，避免把地面当作障碍。
        for geom in body.iter('geom'):
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
        ET.SubElement(
            body,
            'geom',
            name='navigation_envelope',
            type='cylinder',
            size=f'{BODY_RADIUS} {(BODY_TOP-BODY_BOTTOM)/2}',
            pos=f'0 0 {(BODY_TOP+BODY_BOTTOM)/2-WHEEL_RADIUS}',
            contype='1',
            conaffinity='2',
            margin=str(CLEARANCE + 0.001),
            rgba='0 0 0 0',
            group='5',
            mass='0',
        )
        root.find('worldbody').append(body)
        self.model = compile_model(root)

        # MuJoCo 按场景尺寸缩放近裁剪距离；默认值会裁掉近处墙面。
        # 缩短裁剪距离，让画面与实际碰撞位置一致。
        self.model.vis.map.znear = 0.002 / self.model.stat.extent
        self.envelope = self.model.geom('navigation_envelope').id
        base = self.model.body('base_link').id
        self.obstacles = []
        for i in range(self.model.ngeom):
            owner = int(self.model.geom_bodyid[i])
            robot = False
            while owner:
                if owner == base:
                    robot = True
                    break
                owner = int(self.model.body_parentid[owner])
            if not robot and (self.model.geom_contype[i] or self.model.geom_conaffinity[i]):
                self.obstacles.append(i)
                self.model.geom_contype[i] = 2
                self.model.geom_conaffinity[i] = 1
        if not self.obstacles:
            raise RuntimeError('Scene contains no collision geometry')
        self.data = mujoco.MjData(self.model)
        for part in ('left', 'right'):
            joint = self.model.joint(f'{part}_wheel_joint').id
            setattr(self, f'_{part}_joint', joint)
            setattr(self, f'_{part}_qpos', self.model.jnt_qposadr[joint])
            setattr(self, f'_{part}_dof', self.model.jnt_dofadr[joint])
        self._base_joint = self.model.joint('base_joint').id
        self._base_qpos = self.model.jnt_qposadr[self._base_joint]
        self._base_dof = self.model.jnt_dofadr[self._base_joint]
        self._yaw = 0.0
        self.blocked = False
        self.blocked_steps = 0
        self.last_obstacle = ''
        mujoco.mj_forward(self.model, self.data)
        if self._collision():
            raise RuntimeError(f'Robot spawn overlaps obstacle: {self.last_obstacle}')

    def _collision(self):
        for contact in self.data.contact:
            if self.envelope in contact.geom and contact.dist < CLEARANCE:
                other = int(
                    contact.geom[1] if contact.geom[0] == self.envelope else contact.geom[0]
                )
                self.last_obstacle = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                ) or str(other)
                return True
        return False

    def reset(self):
        super().reset()
        self.blocked = False
        self.blocked_steps = 0
        self.last_obstacle = ''
        if self._collision():
            raise RuntimeError('Reset position is obstructed')

    # 每个运动子步都检查碰撞，受阻时只撤回平移
    def step(self, command):
        linear, angular = map(float, command)
        if not all(math.isfinite(v) for v in (linear, angular)):
            linear = angular = 0.0
        linear = max(-1.5, min(1.5, linear))
        angular = max(-3.0, min(3.0, angular))
        steps = max(1, math.ceil(abs(linear) * PHYSICS_DT / MAX_TRANSLATION_STEP))
        dt = PHYSICS_DT / steps
        travelled = 0.0
        self.blocked = False
        self.last_obstacle = ''
        for _ in range(steps):
            previous = self.data.qpos[self._base_qpos : self._base_qpos + 2].copy()
            heading = self._yaw + angular * dt / 2
            self._yaw = math.atan2(
                math.sin(self._yaw + angular * dt), math.cos(self._yaw + angular * dt)
            )
            self.data.qpos[self._base_qpos + 3 : self._base_qpos + 7] = [
                math.cos(self._yaw / 2),
                0,
                0,
                math.sin(self._yaw / 2),
            ]
            if linear and not self.blocked:
                self.data.qpos[self._base_qpos : self._base_qpos + 2] += (
                    linear * dt * np.array([math.cos(heading), math.sin(heading)])
                )
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_collision(self.model, self.data)
                if self._collision():
                    self.data.qpos[self._base_qpos : self._base_qpos + 2] = previous
                    self.blocked = True
                else:
                    travelled += linear * dt
        actual = travelled / PHYSICS_DT
        if self.blocked:
            self.blocked_steps += 1
        left = (actual - angular * WHEEL_TRACK / 2) / WHEEL_RADIUS
        right = (actual + angular * WHEEL_TRACK / 2) / WHEEL_RADIUS
        self.data.qpos[self._left_qpos] += left * PHYSICS_DT
        self.data.qpos[self._right_qpos] += right * PHYSICS_DT
        self.data.qvel.fill(0)
        self.data.qvel[self._base_dof] = actual * math.cos(self._yaw)
        self.data.qvel[self._base_dof + 1] = actual * math.sin(self._yaw)
        self.data.qvel[self._base_dof + 5] = angular
        self.data.qvel[self._left_dof] = left
        self.data.qvel[self._right_dof] = right
        self.data.time += PHYSICS_DT
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        state = super().state()
        return RobotState(
            state.pose,
            state.velocity,
            {
                'collision_guard': True,
                'blocked': self.blocked,
                'blocked_steps': self.blocked_steps,
                'obstacle': self.last_obstacle,
                'body_radius_m': BODY_RADIUS,
                'clearance_m': CLEARANCE,
            },
        )


