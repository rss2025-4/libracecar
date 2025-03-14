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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar

import pytest
import unshare
from pyroute2 import IPDB, IPRoute, NetNS
from pyroute2.ipdb.interfaces import Interface

P = ParamSpec("P")
R = TypeVar("R")

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
    _tp: type[Exception]
    _str: str


def _sub_subproc_fn(f: Callable[P, R], q: Queue, *args: P.args, **kwargs: P.kwargs):
    try:
        ans = f(*args, **kwargs)
        q.put_nowait(_subproc_ret_ok(ans))
    except Exception as e:
        traceback.print_exc()
        q.put(_subproc_err(type(e), str(e)))

    # otherwise queue might not flush and item not visible to parent process
    q.close()
    q.join_thread()
    os._exit(0)


def _subproc_fn(f: Callable[P, R], q: Queue, *args: P.args, **kwargs: P.kwargs):
    setup_isolation()
    p = multiprocessing.Process(
        target=_sub_subproc_fn, args=(f, q, *args), kwargs=kwargs
    )
    p.start()
    p.join()
    sys.exit(p.exitcode)


def run_isolated(f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    q = Queue()
    p = multiprocessing.Process(target=_subproc_fn, args=(f, q, *args), kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError("subprocess failed")
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
    raise error_t(res._str)


def isolate(f: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return run_isolated(f, *args, **kwargs)

    try:
        setattr(inner, "__signature__", inspect.signature(f))
    except:
        pass
    return inner
