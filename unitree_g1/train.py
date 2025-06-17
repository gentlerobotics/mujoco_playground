# Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import flax, flax.serialization as fs

import functools, json, pathlib, datetime, imageio.v3 as iio
import ctypes, os, importlib.metadata as im, jax, pathlib, subprocess, sys

# MuJoCo Playground
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

# Robot Wrapper
from robot import make_robot

class RobotTrainer(env_name = 'G1JoystickFlatTerrain'):
    def __init__(self, env_name):
        self.valid_envs = registry.locomotion.ALL_ENVS
        if env_name not in self.valid_envs:
            raise ValueError(f"Invalid environment name: {env_name}. Valid environments are: {self.valid_envs}")
        self.env = registry.load(env_name)
        self.env_cfg = registry.get_default_config(env_name)

        self.ppo_params = locomotion_params.brax_ppo_config(env_name)

        # Training progress display params
        self.x_data, self.y_data, self.y_dataerr = [], [], []
        self.times = [datetime.now()]
    
    def _progress(self, num_steps, metrics):
        clear_output(wait=True)

        self.times.append(datetime.now())
        self.x_data.append(num_steps)
        self.y_data.append(metrics["eval/episode_reward"])
        self.y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, self.ppo_params["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={self.y_data[-1]:.3f}")
        plt.errorbar(self.x_data, self.y_data, yerr=self.y_dataerr, color="blue")

        display(plt.gcf())

    def train(self, out_dir: str = "checkpoints", run_name: str =None):
        randomizer = registry.get_domain_randomizer(self.env_name)
        ppo_training_params = dict(self.ppo_params)
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in self.ppo_params:
            del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **self.ppo_params.network_factory
        )

        train_fn = functools.partial(
            ppo.train, **dict(ppo_training_params),
            network_factory=network_factory,
            randomization_fn=randomizer,
            progress_fn=self._progress
        )

        make_inference_fn, params, metrics = train_fn(
            environment=self.env,
            eval_env=registry.load(self.env_name, config=self.env_cfg),
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        
        # ---------- SAVE ----------------------------------------------------
        ckpt_dir = pathlib.Path(out_dir) / (run_name or self.env_name)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # a. parameters (PyTree)  — orbax gives nice metadata + CRC
        save_args = ocp.save_args_from_target(params)
        ocp.PyTreeCheckpointer().save(ckpt_dir / "policy", params, save_args=save_args)

        # b. network_factory kwargs so we can rebuild make_inference_fn later
        (ckpt_dir / "network_factory.json").write_text(
            json.dumps(getattr(self.ppo_params, "network_factory", {}), indent=2)
        )

        # c. obs normaliser state if present
        if "normalizer" in params:
            norm_state = flax.serialization.to_bytes(params["normalizer"])
            (ckpt_dir / "obs_norm.msgpack").write_bytes(norm_state)

        # d. final metrics (optional)
        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        # --------------------------------------------------------------------

        return make_inference_fn, params, metrics

         # ---------------------- INFERENCE / ROLLOUT ---------------------------
    def _infer(self, *, ckpt_root: str = "checkpoints", run_name: str | None = None,
              steps: int = 1000, video: str | None = "rollout.mp4"):

        ckpt_dir = pathlib.Path(ckpt_root) / (run_name or self.env_name)
        if not (ckpt_dir / "policy").exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_dir}\n"
                                    "→ run .train() first.")

        # 1. load params & extras
        params = ocp.PyTreeCheckpointer().restore(ckpt_dir / "policy")
        nf_kwargs = json.loads((ckpt_dir / "network_factory.json").read_text())
        norm_state = None
        try:
            norm_state = fs.from_bytes({}, (ckpt_dir / "obs_norm.msgpack").read_bytes())
        except FileNotFoundError:
            pass

        # 2. rebuild + jit policy
        net_factory  = functools.partial(ppo_networks.make_ppo_networks, **nf_kwargs)
        make_inf_fn  = net_factory(is_eval=True).inference_fn
        infer_fn     = jax.jit(make_inf_fn(params, deterministic=True))

        # 3. rollout in env
        env = registry.load(self.env_name)          # fresh copy
        reset, step = map(jax.jit, (env.reset, env.step))
        rng   = jax.random.PRNGKey(123)
        state = reset(rng)

        frames = []
        for t in range(min(steps, self.env_cfg.episode_length)):
            obs = state.obs
            if norm_state is not None:
                mean, var = norm_state["mean"], norm_state["var"]
                obs = (obs - mean) / jp.sqrt(var + 1e-8)

            cmd  = jp.array([1.0, 0.0, 0.0])        # simple forward command
            ctrl, _ = infer_fn(jp.concatenate([obs, cmd]), rng)
            state   = step(state, ctrl)
            if state.done: break
            frames.append(env.render([state])[0])

        # 4. save / show video
        if video:
            fps = 1.0 / env.dt
            iio.imwrite(video, np.stack(frames), fps=fps, codec="libx264")
            print(f"✓ video saved → {video}  ({len(frames)/fps:.1f}s)")

        return frames
    
    def infer(
        self,
        *,                                  # keyword-only
        backend: str = "sim",               #  "sim"  or  "real"
        interface: str = "enp6s0",          #  NIC for DDSRobot
        steps: int = 1_000,                 #  sim steps | real-time cycles
        ckpt_root: str = "checkpoints",
        run_name: str | None = None,
        video: str | None = "rollout.mp4",  #  only written in sim mode
    ) -> list[np.ndarray] | None:
        """
        Roll out the saved policy either in MuJoCo ('sim') or on the real G1
        ('real' → DDS back-end).

        Returns:
            list of RGB frames if backend=='sim' and video is None, otherwise None.
        Raises:
            FileNotFoundError if the requested checkpoint is missing.
        """
        # ── 0.  checkpoint ----------------------------------------------------
        ckpt_dir = pathlib.Path(ckpt_root) / (run_name or self.env_name)
        if not (ckpt_dir / "policy").exists():
            raise FileNotFoundError(f"no checkpoint at {ckpt_dir} – run .train() first")

        params = ocp.PyTreeCheckpointer().restore(ckpt_dir / "policy")
        nf_kwargs = json.loads((ckpt_dir / "network_factory.json").read_text())
        try:
            norm_state = fs.from_bytes({}, (ckpt_dir / "obs_norm.msgpack").read_bytes())
        except FileNotFoundError:
            norm_state = None

        # ── 1.  rebuild + JIT policy -----------------------------------------
        net_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_kwargs)
        make_inf_fn = net_factory(is_eval=True).inference_fn
        infer_fn    = jax.jit(make_inf_fn(params, deterministic=True))

        # ── 2.  choose back-end ----------------------------------------------
        if backend == "sim":
            env   = registry.load(self.env_name)
            robot = make_robot("sim", env=env)          # SimRobot
        elif backend == "real":
            print("⚠  Real G1 selected – keep an e-stop nearby!")
            input("Press ENTER to enable torque control…")
            robot = make_robot("dds", iface=interface, sim=False)  # DDSRobot
        else:
            raise ValueError("backend must be 'sim' or 'real'")

        robot.reset()
        rng     = jax.random.PRNGKey(0)
        frames  = []                        # only filled in sim

        # ── 3.  main control loop --------------------------------------------
        try:
            for t in range(steps if backend == "sim" else 10**9):
                if not robot.ok():
                    print("robot reported not-ok → stopping")
                    break

                obs = robot.get_observation()
                if norm_state is not None:
                    mean, var = norm_state["mean"], norm_state["var"]
                    obs = (obs - mean) / np.sqrt(var + 1e-8)

                cmd = np.array([1.0, 0.0, 0.0], np.float32)        # ← TODO joystick
                act, _ = infer_fn(np.concatenate([obs, cmd]), rng)
                robot.send_action(np.asarray(act, np.float32))
                robot.sleep()

                if backend == "sim":
                    frames.append(env.render([robot._state])[0])   # pylint:disable=protected-access
                    if robot._state.done:                          # idem
                        break
        except KeyboardInterrupt:
            print("✋ Interrupted by user")

        # ── 4.  optional video output (sim only) ------------------------------
        if backend == "sim" and video:
            fps = 1.0 / env.dt
            iio.imwrite(video, np.stack(frames), fps=fps, codec="libx264")
            print(f"✓ video saved → {video}  ({len(frames)/fps:.1f}s, {fps:.0f} fps)")
            return None

        return frames if backend == "sim" else None

    
    
    