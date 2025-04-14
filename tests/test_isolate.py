import pytest

from libracecar.sandbox import isolate


@isolate
def test_isolate():
    pass


def test_isolate2():

    @isolate
    def inner():
        return "abc"

    assert inner() == "abc"


def test_isolate3():

    @isolate
    def inner():
        assert False

    with pytest.raises(Exception):
        inner()
