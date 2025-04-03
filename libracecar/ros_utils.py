from builtin_interfaces.msg import Time


def time_msg_to_float(t: Time | float) -> float:
    if isinstance(t, Time):
        return t.sec + t.nanosec / 1e9
    assert isinstance(t, float)
    return t


def float_to_time_msg(t: float) -> Time:
    time_msg = Time()
    time_msg.sec = int(t)
    time_msg.nanosec = int((t - time_msg.sec) * 1e9)
    return time_msg
