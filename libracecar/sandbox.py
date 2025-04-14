"""
ROS uses network to communicate between nodes, in a distributed fashion.

This module aims to run some ros nodes in an isolated fasion. It makes sure:

1) isolated ros nodes does not see nodes running outside isolation

2) isloated ros nodes are always killed when we finish

This modules uses linux namespaces to do isolation;
You can read more about that `here <https://thomasvanlaere.com/posts/2020/12/exploring-containers-part-3/>`_
(disclaimer: found on google, didnt read myself)

the main entrypoint of the module is ``isolate``.
Here is an example of running bash under isolate:

.. code-block:: python

    import subprocess
    from libracecar.sandbox import isolate

    @isolate
    def main():
        subprocess.run("bash -i", shell=True, check=True)
        return "finished!"

    if __name__ == "__main__":
        assert main() == "finished!"

if you run ``ros2 topic list`` under this bash session,
you will find that nodes outside isolation is not visible.

here is a more complete example:

.. code-block:: python

    import subprocess
    import time

    from libracecar.sandbox import isolate
    from libracecar.test_utils import proc_manager


    # indicate success with an exception
    class _ok(BaseException):
        pass


    def wait_time():
        time.sleep(5)
        raise _ok()


    @isolate
    def main():
        procs = proc_manager.new()

        # run some stuff
        procs.popen(["rviz2"])
        procs.ros_launch("racecar_simulator", "simulate.launch.xml")
        procs.thread(wait_time)

        # run some gui apps so that you can inspect what is happening in a subshell
        # only tested with emacs
        procs.popen(["emacs"])

        # wait until anything fails/throws, then fail
        try:
            procs.spin()
        except _ok:
            return "success!"


    if __name__ == "__main__":
        main()

it appears that pytest can access local varaibles under ``isolate``; i have no idea why this works though.

.. code-block:: python

    import pytest
    from libracecar.sandbox import isolate

    @isolate
    def test_me(x=2):
        assert x < 0

.. code-block:: text

    ========================================================== FAILURES ==========================================================
    __________________________________________________________ test_me ___________________________________________________________

    E   AssertionError: (from <function test_me at 0x7f355e9e0e50> under isolate):
        assert 2 < 0

known problems:

type of exception is not kept; instead an exception with the same name is thrown.

i have been unable to make this work under github actions
"""

import ctypes
import ctypes.util
import functools
import inspect
import multiprocessing
import os
import sys
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import Any, Callable, ParamSpec, TypeVar, Union

import unshare
from better_exceptions import format_exception
from pyroute2 import IPRoute
from typing_extensions import Never

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

    Path("/proc/self/uid_map").write_text(f"{uid} {uid} 1\n")
    Path("/proc/self/setgroups").write_text(f"deny")
    Path("/proc/self/gid_map").write_text(f"{gid} {gid} 1\n")

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


def _close_queue_and_exit(q: Queue, code: int) -> Never:
    q.close()
    q.join_thread()
    os._exit(code)


_global_q: Union[Queue, None] = None


def throw(e: BaseException) -> Never:
    assert _global_q is not None
    return _throw(_global_q, e)


def _throw(q: Queue, e: BaseException) -> Never:
    try:
        # exc_type, exc_value, exc_tb = sys.exc_info()
        exc_type, exc_value, exc_tb = type(e), e, e.__traceback__
        ex_fmt = "".join(format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(ex_fmt)
        sys.stderr.flush()

        q.put_nowait(_subproc_err(type(e), str(e)))
        _close_queue_and_exit(q, 0)
    except BaseException as e2:
        sys.stderr.write(f"_run_user_fn: failed during exception handling: {e2}")
        sys.stdout.flush()
    finally:
        os._exit(1)


def _run_user_fn(f: Callable[P, R], q: Queue, *args: P.args, **kwargs: P.kwargs):
    # runs f, pushes at most one value or exceptions to q
    #
    # usually:
    #   exits with 0 if pushed, otherwise nonzero
    # promises:
    #   exit 0 imply value pushed

    # nvidia gpus wants to have the "right" /proc
    mount("proc", "/proc", "proc", "")

    global _global_q
    _global_q = q

    try:
        __tracebackhide__ = True
        ans = f(*args, **kwargs)
        q.put_nowait(_subproc_ret_ok(ans))
        _close_queue_and_exit(q, 0)
    except BaseException as e:
        _throw(q, e)
    finally:
        os._exit(1)


def _namespace_root(f: Callable[P, R], q: Queue, *args: P.args, **kwargs: P.kwargs):
    p = None
    try:
        setup_isolation()
        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(target=_run_user_fn, args=(f, q, *args), kwargs=kwargs)
        p.start()
        p.join()
    except BaseException as e:
        if p is not None and p.exitcode is not None and p.exitcode != 0:
            sys.exit(p.exitcode)
        q.put_nowait(_subproc_err(type(e), str(e)))
        _close_queue_and_exit(q, 0)
    finally:
        os._exit(1)


def run_isolated(f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    # communicating with _namespace_root via Queue can only work via a fork
    # if we used spawn than Queue is over network which we will disconnect
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_namespace_root, args=(f, q, *args), kwargs=kwargs)
    e = None
    try:
        p.start()
        p.join()
    except BaseException as e_:
        e = e_
        p.kill()

    try:
        res = q.get_nowait()
    except Empty:
        if e is not None:
            raise e from None
        res = _subproc_err(RuntimeError, f"unknown error; exitcode={p.exitcode}")

    if isinstance(res, _subproc_ret_ok):
        return res.val
    assert isinstance(res, _subproc_err)

    class error_t(Exception):
        pass

    error_t.__name__ = res._tp.__name__
    error_t.__module__ = res._tp.__module__
    error_t.__qualname__ = res._tp.__qualname__

    __tracebackhide__ = True
    raise error_t(
        f"(from {f} under isolate):\n"
        + "\n".join("> " + x for x in res._str.splitlines())
    )


def isolate(f: Callable[P, R]) -> Callable[P, R]:
    """
    runs ``f`` in its own linux namespace.

    Propagates return value and exceptions.

    The return type ``R`` is required to be pickleable.

    Type of Exception is NOT kept; instead an exception with the same name is thrown.
    """

    @functools.wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return run_isolated(f, *args, **kwargs)

    try:
        setattr(inner, "__signature__", inspect.signature(f))
    except:
        pass
    return inner
