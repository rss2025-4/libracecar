import threading
from queue import Queue
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

import jax
import numpy as np
from jax import lax
from jax import tree_util as jtu

from .utils import (
    PropagatingThread,
    cast,
    cond_,
    debug_callback,
    debug_print,
    flike,
    fval,
    io_callback_,
    jit,
    timer,
    tree_to_ShapeDtypeStruct,
)

P = ParamSpec("P")
T = TypeVar("T")
A = TypeVar("A")
R = TypeVar("R")


class dispatch_spec(Generic[T]):
    def __init__(
        self,
        fn: Callable[Concatenate[T, P], tuple[T, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        self.fn = fn
        self.arg_ex = tree_to_ShapeDtypeStruct((args, kwargs))


class jax_jit_dispatcher(Generic[T]):

    def __init__(self, *methods: dispatch_spec[T]):
        self.methods = methods

        self.lock = threading.Lock()

        self.requesttype_q: Queue[int] = Queue()
        self.request_q = Queue()
        self.response_q = Queue()

    def process(
        self,
        fn: Callable[Concatenate[T, P], tuple[T, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        with self.lock:
            for i, meth in enumerate(self.methods):
                if meth.fn is fn:
                    self.requesttype_q.put_nowait(i)
                    self.request_q.put_nowait((args, kwargs))
                    return self.response_q.get()
        assert False

    def run_with_setup(
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> PropagatingThread:
        lowered = jit(lambda: self.spin(fn(*args, **kwargs))).lower()
        print("compiling:")
        with timer.create() as t:
            comp = lowered.compile()
            print(f"took {t.val}s")
            print(comp.cost_analysis())

        def thread_fn():
            _ = comp()
            assert False

        ans = PropagatingThread(target=thread_fn)
        ans.start()
        return ans

    def run(self, init: T) -> PropagatingThread:
        return self.run_with_setup(lambda: init)

    def _requesttype_callback(self):
        return self.requesttype_q.get()

    def _request_callback(self):
        return self.request_q.get()

    def _response_callback(self, res):
        self.response_q.put_nowait(res)

    def spin(self, init_state: T):

        def handle_request(reqtype: int, s: T) -> T:
            meth = self.methods[reqtype]
            req_args, req_kwargs = io_callback_(self._request_callback, meth.arg_ex)()
            new_s, ans = meth.fn(s, *req_args, **req_kwargs)
            io_callback_(self._response_callback, None)(ans)
            return new_s

        def loop_fn(s: T):
            reqtype = io_callback_(
                self._requesttype_callback, jax.ShapeDtypeStruct((), np.int32)
            )()

            for i in range(len(self.methods)):
                s = cond_(
                    reqtype == i,
                    true_fun=lambda: handle_request(i, s),
                    false_fun=lambda: s,
                )

            return s

        lax.while_loop(
            cond_fun=lambda _: True,
            body_fun=loop_fn,
            init_val=init_state,
        )


def divide_x_at_zero(f: Callable[[fval], T]) -> Callable[[fval], T]:
    """assumes f(0.) = 0. result only have valid first order gradient."""

    def jvp(f: Callable[[fval], T]) -> Callable[[flike], T]:
        def inner(x: flike):
            _, tangents = jax.jvp(f, primals=(x,), tangents=(1.0,))
            return tangents

        return inner

    def inner(x: fval) -> T:
        return jtu.tree_map(lambda a, b: a + b / 2 * x, jvp(f)(0.0), jvp(jvp(f))(0.0))

    return inner
