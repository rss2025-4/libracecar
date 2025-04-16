import threading
from queue import Queue
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

import jax
import numpy as np
import numpyro
from jax import Array, lax, random
from jax import tree_util as jtu
from jax.experimental.checkify import Error, ErrorCategory, checkify, user_checks

from .utils import (
    PropagatingThread,
    cond_,
    flike,
    fval,
    io_callback_,
    jit,
    timer,
    tree_select,
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
        seed: bool,
        checks: frozenset[ErrorCategory],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        self.seed = seed
        self.checks = checks
        self.fn = fn
        self.arg_ex = tree_to_ShapeDtypeStruct((args, kwargs))

    def _call(self, state: T, *args, **kwargs):
        err, (new_state, ans) = checkify(self.fn, self.checks)(state, *args, **kwargs)
        print("potential errors:", err._pred)
        is_err = False
        for x in err._pred.values():
            is_err |= x
        print("is_err", is_err)
        if is_err is not False:
            new_state = tree_select(is_err, on_true=state, on_false=new_state)
        return new_state, (err, ans)


def dispatch(
    fn: Callable[Concatenate[T, P], tuple[T, R]],
    *,
    seed: bool = True,
    checks: frozenset[ErrorCategory] = user_checks,
) -> Callable[P, dispatch_spec[T]]:
    def inner(*args: P.args, **kwargs: P.kwargs):
        return dispatch_spec(fn, seed, checks, *args, **kwargs)

    return inner


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
                    err, ans = self.response_q.get()
                    assert isinstance(err, Error)
                    err.throw()
                    return ans

        assert False

    def run_with_setup(
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> PropagatingThread:
        def run_fn(*args: P.args, **kwargs: P.kwargs):
            return self.spin(fn(*args, **kwargs))

        lowered = jit(run_fn).lower(*args, **kwargs)
        print("compiling:")
        with timer.create() as t:
            comp = lowered.compile()
            print(f"took {t.val}s")
            print(comp.cost_analysis())

        def thread_fn():
            _ = comp(*args, **kwargs)
            assert False

        ans = PropagatingThread(target=thread_fn)
        ans.start()
        return ans

    def run(self, init: T) -> PropagatingThread:
        return self.run_with_setup(lambda x: x, init)

    def _requesttype_callback(self):
        return self.requesttype_q.get()

    def _request_callback(self):
        return self.request_q.get()

    def _response_callback(self, res):
        self.response_q.put_nowait(res)

    def spin(self, init_state: T):

        init_seed = random.PRNGKey(0)

        def handle_request(reqtype: int, s: tuple[T, Array]) -> tuple[T, Array]:
            state, rng = s
            meth = self.methods[reqtype]
            req_args, req_kwargs = io_callback_(self._request_callback, meth.arg_ex)()

            if meth.seed:
                rng, this_key = random.split(rng)
                assert isinstance(rng, Array)
                with numpyro.handlers.seed(rng_seed=this_key):
                    new_s, ans = meth._call(state, *req_args, **req_kwargs)
            else:
                new_s, ans = meth._call(state, *req_args, **req_kwargs)

            io_callback_(self._response_callback, None)(ans)
            return new_s, rng

        def loop_fn(s: tuple[T, Array]):
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
            init_val=(init_state, init_seed),
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
