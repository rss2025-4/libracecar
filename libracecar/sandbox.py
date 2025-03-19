import ctypes
import ctypes.util
import functools
import inspect
import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, Protocol, TypeVar

import pytest
import unshare
from pyroute2 import IPDB, IPRoute, NetNS
from pyroute2.ipdb.interfaces import Interface

P = ParamSpec("P")
R = TypeVar("R", covariant=True)

# https://stackoverflow.com/questions/1667257/how-do-i-mount-a-filesystem-using-python
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mount.argtypes = (
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_ulong,
    ctypes.c_char_p,
)


def mount(source: str, target: str, fs: str, options=""):
    ret = libc.mount(source.encode(), target.encode(), fs.encode(), 0, options.encode())
    if ret < 0:
        errno = ctypes.get_errno()
        raise OSError(
            errno,
            f"Error mounting {source} ({fs}) on {target} with options '{options}': {os.strerror(errno)}",
        )


def setup_isolation() -> None:
    uid = os.getuid()
    gid = os.getgid()

    unshare.unshare(
        0
        | unshare.CLONE_NEWIPC
        | unshare.CLONE_NEWNET
        | unshare.CLONE_NEWNS
        | unshare.CLONE_NEWPID
        | unshare.CLONE_NEWUSER
    )

    Path("/proc/self/uid_map").write_text(f"0 {uid} 1\n")
    Path("/proc/self/setgroups").write_text(f"deny")
    Path("/proc/self/gid_map").write_text(f"0 {gid} 1\n")

    mount("tmpfs", "/dev/shm", "tmpfs", "")

    ip = IPRoute()
    idxs = ip.link_lookup(ifname="lo")
    assert isinstance(idxs, list) and len(idxs) == 1
    ip.link("set", index=idxs[0], state="up")


@dataclass
class _subproc_ret_ok:
    val: Any


@dataclass
class _subproc_err:
    _tp: type[BaseException]
    _str: str


def _run_user_fn(f: Callable[P, R], q: Queue, *args: P.args, **kwargs: P.kwargs):
    # runs f, and push exception string or return value into a queue
    try:
        __tracebackhide__ = True
        ans = f(*args, **kwargs)
        q.put_nowait(_subproc_ret_ok(ans))
    except BaseException as e:
        traceback.print_exc()
        q.put_nowait(_subproc_err(type(e), str(e)))


def _namespace_root(f: Callable[P, R], q: Queue, *args: P.args, **kwargs: P.kwargs):
    setup_isolation()
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_run_user_fn, args=(f, q, *args), kwargs=kwargs)
    try:
        p.start()
        p.join()
    except BaseException as e:
        if p.exitcode is not None and p.exitcode != 0:
            sys.exit(p.exitcode)
        traceback.print_exc()
        q.put_nowait(_subproc_err(type(e), str(e)))
        sys.exit(0)


def run_isolated(f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    # communicating with _namespace_root via Queue can only work via a fork
    # if we used spawn than Queue is over network which we will disconnect
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_namespace_root, args=(f, q, *args), kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{f} under isolate failed")
    res = q.get_nowait()
    if isinstance(res, _subproc_ret_ok):
        return res.val
    assert isinstance(res, _subproc_err)

    class error_t(Exception):
        pass

    error_t.__name__ = res._tp.__name__
    error_t.__module__ = res._tp.__module__
    error_t.__qualname__ = res._tp.__qualname__

    __tracebackhide__ = True
    raise error_t(f"(from {f} under isolate):\n" + res._str)


def isolate(f: Callable[P, R]) -> Callable[P, R]:
    # f is required to be pickleable
    @functools.wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return run_isolated(f, *args, **kwargs)

    try:
        setattr(inner, "__signature__", inspect.signature(f))
    except:
        pass
    return inner
