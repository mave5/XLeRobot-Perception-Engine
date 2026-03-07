from xlerobot_personality.tracking_controller import PIDController


def test_pid_output_sign():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, integral_limit=10.0)
    out = pid.update(error=3.0, dt=0.1)
    assert out > 0


def test_pid_integral_clamp():
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, integral_limit=2.0)
    for _ in range(100):
        pid.update(error=10.0, dt=0.1)
    out = pid.update(error=10.0, dt=0.1)
    assert out <= 2.0
