import multiprocessing
import subprocess
import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Callable, Concatenate, ParamSpec, Protocol, TypeVar

import rclpy
from rclpy.node import Node

from .sandbox import run_isolated
from .utils import PropagatingThread, cast, cast_unchecked

P = ParamSpec("P")
R = TypeVar("R")


class _poll(Protocol):
    def poll(self): ...


@dataclass
class _proc:
    p: subprocess.Popen

    def poll(self):
        return_code = self.p.poll()
        if return_code is not None and return_code != 0:
            raise subprocess.CalledProcessError(returncode=return_code, cmd=self.p.args)


@dataclass
class _call:
    p: multiprocessing.Process

    def poll(self):
        if self.p.exitcode is not None and self.p.exitcode != 0:
            raise RuntimeError(f"{self.p} failed")


@dataclass
class _thread:
    t: PropagatingThread

    def poll(self):
        if not self.t.is_alive() and self.t.exc:
            raise self.t.exc


def _run_node(f: Callable[P, Node], *args: P.args, **kwargs: P.kwargs):
    rclpy.init()
    rclpy.spin(f(*args, **kwargs))
    rclpy.shutdown()


@dataclass
class proc_manager:
    _parts: list[_poll]

    @staticmethod
    def new() -> "proc_manager":
        return proc_manager([])

    @property
    def popen(self):
        def inner(*args, **kwargs):
            x = subprocess.Popen(*args, **kwargs)
            self._parts.append(_proc(x))
            return x

        return cast_unchecked(subprocess.Popen)(inner)

    def call(
        self, f: Callable[P, None], *args: P.args, **kwargs: P.kwargs
    ) -> multiprocessing.Process:
        p = multiprocessing.Process(target=f, args=args, kwargs=kwargs)
        p.start()
        self._parts.append(_call(p))
        return p

    def thread(self, f: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> Thread:
        t = PropagatingThread(target=f, args=args, kwargs=kwargs)
        t.start()
        self._parts.append(_thread(t))
        return t

    def ros_launch(self, package_name: str, launch_file_name: str):
        # WARNING: this currently silently fails if a node in the lanuch file fails
        return self.popen(["ros2", "launch", str(package_name), str(launch_file_name)])

    def ros_run(self, package_name: str, executable_name: str):
        return self.popen(["ros2", "run", str(package_name), str(executable_name)])

    def ros_node_subproc(
        self, node_t: Callable[P, Node], *args: P.args, **kwargs: P.kwargs
    ):
        return self.call(_run_node, node_t, *args, **kwargs)

    def ros_node_thread(
        self, node_t: Callable[P, Node], *args: P.args, **kwargs: P.kwargs
    ):
        return self.thread(_run_node, node_t, *args, **kwargs)

    def spin(self):
        while True:
            for x in self._parts:
                x.poll()
            time.sleep(0.1)
