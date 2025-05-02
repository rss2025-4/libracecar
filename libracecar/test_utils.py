import multiprocessing
import os
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.context import SpawnProcess
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Lock, Thread
from typing import (
    Callable,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
)

import rclpy
import tf2_ros
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.subscription import Subscription
from rclpy.time import Time

from .sandbox import print_ex, throw
from .utils import PropagatingThread, cast_unchecked

P = ParamSpec("P")
R = TypeVar("R")

T = TypeVar("T")
N = TypeVar("N", bound=Node)


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
    p: BaseProcess

    def poll(self):
        if self.p.exitcode is not None and self.p.exitcode != 0:
            raise RuntimeError(f"{self.p} failed")


@dataclass
class _thread:
    t: PropagatingThread

    def poll(self):
        if not self.t.is_alive() and self.t.exc:
            raise self.t.exc


@dataclass
class rclpy_config:
    params_file: Path | None = None

    def as_args(self):
        args: list[str] = []
        if self.params_file is not None:
            assert self.params_file.exists()
            args.append("--ros-args")
            args.append("--params-file")
            args.append(str(self.params_file))
        return args


def _run_node_subproc(f: Callable[[], Node], ros_args: list[str], args, kwargs):
    rclpy.init(args=ros_args)
    rclpy.spin(f(*args, **kwargs))
    rclpy.shutdown()


class GlobalNode(Node):
    def __init__(self, context: Context) -> None:
        super().__init__(type(self).__qualname__, context=context)
        self.lock = Lock()

        self.__pubs: dict[str, Publisher] = {}

        self.__subs: dict[str, tuple[Subscription, Queue]] = {}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def lookup_transform(
        self, target_frame: str, source_frame: str
    ) -> tf2_ros.TransformStamped:
        with self.lock:
            return self.tf_buffer.lookup_transform(
                target_frame=target_frame, source_frame=source_frame, time=Time()
            )

    def publish(self, topic: str, val):
        with self.lock:
            if topic not in self.__pubs:
                self.__pubs[topic] = self.create_publisher(
                    type(val),
                    topic,
                    qos_profile=QoSProfile(
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        history=QoSHistoryPolicy.KEEP_ALL,
                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    ),
                )
            assert self.__pubs[topic].msg_type is type(val)
            self.__pubs[topic].publish(val)

    def read_queue(
        self,
        topic: str,
        tp: type[T],
        *,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    ) -> "Queue[T]":
        with self.lock:
            if topic not in self.__subs:
                q = Queue()
                sub = self.create_subscription(
                    tp,
                    topic,
                    lambda msg: q.put_nowait(msg),
                    qos_profile=QoSProfile(
                        reliability=reliability,
                        history=QoSHistoryPolicy.KEEP_ALL,
                        durability=durability,
                    ),
                )
                self.__subs[topic] = sub, q
            sub, q = self.__subs[topic]
            assert sub.msg_type is tp

        return q


@dataclass
class proc_manager:
    _parts: list[_poll]
    _globalnode: Union[GlobalNode, None] = None

    @staticmethod
    def new() -> "proc_manager":
        return proc_manager([])

    @property
    def popen(self):
        def inner(*args, **kwargs):
            # note:
            # you might get this here:
            # /usr/lib/python3.10/subprocess.py:1796: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
            # which is harmless since the subprocess immediately exec to something else
            x = subprocess.Popen(*args, **kwargs, preexec_fn=os.setsid)
            self._parts.append(_proc(x))
            return x

        return cast_unchecked(subprocess.Popen)(inner)

    @staticmethod
    def _do_call(f, args, kwargs):
        try:
            f(*args, **kwargs)
            os._exit(0)
        except BaseException as e:
            print_ex(e)
        finally:
            os._exit(1)

    def call(
        self, f: Callable[P, None], *args: P.args, **kwargs: P.kwargs
    ) -> SpawnProcess:
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=proc_manager._do_call, args=(f, args, kwargs))
        p.start()
        return p

    def thread(self, f: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> Thread:
        t = PropagatingThread(target=f, args=args, kwargs=kwargs, daemon=True)
        t.start()
        self._parts.append(_thread(t))
        return t

    def ros_launch(self, package_name: str, launch_file_name: str, *cmd_args: str):
        # WARNING: this currently silently fails if a node in the lanuch file fails
        return self.popen(
            ["ros2", "launch", str(package_name), str(launch_file_name), *cmd_args]
        )

    def ros_run(
        self,
        package_name: str,
        executable_name: str,
        *,
        ros_params: dict[str, str] | None = None,
        params_file: Path | str | None = None,
    ):
        cmd = ["ros2", "run", str(package_name), str(executable_name), "--ros-args"]

        if ros_params is not None:
            for x, y in ros_params.items():
                cmd.append("-p")
                cmd.append(f"{x}:={y}")

        if params_file is not None:
            cmd.append("--params-file")
            cmd.append(str(params_file))

        return self.popen(cmd)

    def ros_node_subproc(self, node_t: Callable[P, Node], cfg=rclpy_config()):
        def inner(*args: P.args, **kwargs: P.kwargs):
            return self.call(_run_node_subproc, node_t, cfg.as_args(), args, kwargs)

        return inner

    def _ros_node_thread(self, context: Context, executor: SingleThreadedExecutor):
        try:
            executor.spin()
        finally:
            context.shutdown()

    def ros_node_thread(self, node_fn: Callable[[Context], N], cfg=rclpy_config()) -> N:
        context = Context()
        context.init(args=cfg.as_args())

        node = node_fn(context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)

        self.thread(self._ros_node_thread, context, executor)

        return node

    def spin(self):
        try:
            while True:
                for x in self._parts:
                    x.poll()
                time.sleep(0.1)
        except BaseException as e:
            throw(e)

    def spin_thread(self):
        self.thread(self.spin)

    def _ensure_globalnode(self):
        if self._globalnode is None:
            self._globalnode = self.ros_node_thread(GlobalNode)
        return self._globalnode

    def publish(self, topic: str, val):
        self._ensure_globalnode().publish(topic, val)

    def read(
        self,
        topic: str,
        tp: type[T],
        timeout: float | None = None,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    ) -> T:
        q = self._ensure_globalnode().read_queue(
            topic, tp, reliability=reliability, durability=durability
        )
        return q.get(timeout=timeout)

    def lookup_transform(
        self, target_frame: str, source_frame: str
    ) -> tf2_ros.TransformStamped:
        return self._ensure_globalnode().lookup_transform(
            target_frame=target_frame, source_frame=source_frame
        )
