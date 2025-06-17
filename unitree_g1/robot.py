"""
Back-end abstraction for Unitree G1 control.

• DDSRobot  – real hardware via unitree_sdk2py (lowcmd / lowstate DDS)
• SimRobot – Brax/MJX environment
"""

from __future__ import annotations
import abc, time, itertools, pathlib
import numpy as np, jax.numpy as jp

# ───────────────────────── 0.  helper constants ───────────────────────────
G1_NUM_MOTOR = 29
Kp = np.array([60,60,60,100,40,40, 60,60,60,100,40,40, 60,40,40,
               40,40,40,40,40,40,40, 40,40,40,40,40,40,40], np.float32)
Kd = np.ones_like(Kp)

# motor enable flag (1 = enable torque / position control)
MOTOR_ENABLE = 1

# ───────────────────────── 1.  base interface ─────────────────────────────
class BaseRobot(abc.ABC):
    def __init__(self, hz: int):
        self._dt = 1.0 / hz
    def sleep(self): time.sleep(self._dt)
    @abc.abstractmethod
    def reset(self): ...
    @abc.abstractmethod
    def ok(self) -> bool: ...
    @abc.abstractmethod
    def get_observation(self) -> np.ndarray: ...
    @abc.abstractmethod
    def send_action(self, act: np.ndarray): ...

# ───────────────────────── 2.  simulation back-end ────────────────────────
class SimRobot(BaseRobot):
    def __init__(self, env, hz: int | None = None):
        self._env = env
        self._step = env.step
        self._reset = env.reset
        super().__init__(hz or int(1 / env.dt))
        self.reset()
    def reset(self):
        self._state = self._reset(jp.random_key(0))
    def ok(self): return True
    def get_observation(self):
        return np.asarray(self._state.obs, np.float32)
    def send_action(self, act: np.ndarray):
        self._state = self._step(self._state, jp.asarray(act, jp.float32))

# ───────────────────────── 3.  DDS (real robot) back-end ──────────────────
try:
    from unitree_sdk2py.core.channel import (
        ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
except ModuleNotFoundError:
    ChannelPublisher = None   # allows import on sim-only machines

class DDSRobot(BaseRobot):
    """Direct DDS control using unitree_sdk2py."""
    def __init__(self, iface="enp6s0", hz=500, sim=False):
        if ChannelPublisher is None:
            raise RuntimeError("unitree_sdk2py not installed")
        super().__init__(hz)
        # 0 → real, 1 → sim (per Unitree docs)
        ChannelFactoryInitialize(int(sim), iface)
        self._crc = CRC()
        # publisher / subscriber
        self._pub  = ChannelPublisher("rt/lowcmd", LowCmd_);  self._pub.Init()
        self._sub  = ChannelSubscriber("rt/lowstate", LowState_)
        self._state: LowState_ | None = None
        self._sub.Init(self._state_cb, 10)
        # build zeroed cmd msg
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self.reset()
    def _state_cb(self, msg):  self._state = msg
    def reset(self):           pass                      # leave robot standing
    def ok(self):              return self._state is not None
    # -------------------- observation -------------------------------------
    def get_observation(self) -> np.ndarray:
        s = self._state
        if s is None:
            return np.zeros(51, np.float32)
        imu = np.array([*s.imu_state.rpy,
                        *s.imu_state.gyroscope,
                        *s.imu_state.accelerometer], np.float32)
        q  = np.array(s.motor_state.q , np.float32)
        dq = np.array(s.motor_state.dq, np.float32)
        return np.concatenate([imu, q[:12], dq[:12]])    # 51-D vector
    # -------------------- action ------------------------------------------
    def send_action(self, act: np.ndarray):
        """act: length-12 target joint torque or position (depending on policy)"""
        for i, a in enumerate(act):
            m = self._cmd.motor_cmd[i]
            m.mode = MOTOR_ENABLE
            m.q    = 0.0            # assuming torque policy
            m.dq   = 0.0
            m.tau  = float(a)
            m.kp, m.kd = float(Kp[i]), float(Kd[i])
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

# ───────────────────────── 4.  factory helper ─────────────────────────────
def make_robot(kind="sim", **kw) -> BaseRobot:
    if kind == "sim":
        env = kw.pop("env")
        return SimRobot(env, **kw)
    if kind == "dds":
        return DDSRobot(**kw)
    raise ValueError(f"unknown back-end {kind!r}")
