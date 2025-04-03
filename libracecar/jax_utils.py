import threading
from queue import Queue
from typing import Callable, Generic, ParamSpec, TypeVar

import jax
import numpy as np
from jax import lax

from .utils import cond_, io_callback_, jit, tree_to_ShapeDtypeStruct

P = ParamSpec("P")
T = TypeVar("T")
A = TypeVar("A")
R = TypeVar("R")


class dispatch_spec(Generic[T]):
    def __init__(self, fn: Callable[[T, A], tuple[T, R]], arg_ex: A):
        self.fn = fn
        self.arg_ex = tree_to_ShapeDtypeStruct(arg_ex)


class jax_jit_dispatcher(Generic[T]):

    def __init__(self, *methods: dispatch_spec[T]):
        self.methods = methods

        self.lock = threading.Lock()

        self.requesttype_q: Queue[int] = Queue()
        self.request_q = Queue()
        self.response_q = Queue()

    def process(self, fn: Callable[[T, A], tuple[T, R]], arg: A) -> R:
        with self.lock:
            for i, meth in enumerate(self.methods):
                if meth.fn is fn:
                    self.requesttype_q.put_nowait(i)
                    self.request_q.put_nowait(arg)
                    return self.response_q.get()
        assert False

    def jit_with_setup(self, setup_fn: Callable[P, T]):
        @jit
        def inner(*args: P.args, **kwargs: P.kwargs):
            init_val = setup_fn(*args, **kwargs)
            self.spin(init_val)

        return inner

    def _requesttype_callback(self):
        return self.requesttype_q.get()

    def _request_callback(self):
        return self.request_q.get()

    def _response_callback(self, res):
        self.response_q.put_nowait(res)

    def spin(self, init_state: T):

        def handle_request(reqtype: int, s: T) -> T:
            meth = self.methods[reqtype]
            req = io_callback_(self._request_callback, meth.arg_ex)()
            new_s, ans = meth.fn(s, req)
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
        assert False
