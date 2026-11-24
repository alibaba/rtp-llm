"""Time and evidence gates for pressure phase transitions."""


class PressurePhases:
    def __init__(self, config, started_s):
        self.config = config
        self.phase = "fill"
        self.entered = started_s
        self.measurement_start = None
        self.measurement_end = None
        self.failure = None

    def fail(self, reason):
        if self.failure is None:
            self.failure = reason
        self.phase = "drain"

    def advance(
        self, now_s, *, fill_ready=False, steady_covered_s=0.0, profile_pending=False
    ):
        if self.phase in ("fill", "steady"):
            timeout = (
                self.config.fill_timeout_s
                if self.phase == "fill"
                else self.config.steady_timeout_s
            )
            if now_s - self.entered >= timeout:
                self.fail("pressure_not_reached")
            elif self.phase == "fill" and fill_ready:
                self.phase, self.entered = "steady", now_s
            elif self.phase == "steady" and steady_covered_s >= self.config.steady_s:
                self.phase = "measurement"
                self.measurement_start = now_s
                self.measurement_end = now_s + self.config.measure_s
        elif self.phase == "measurement" and now_s >= self.measurement_end:
            self.phase = "support" if profile_pending else "drain"
        if self.phase == "support":
            if not profile_pending:
                self.phase = "drain"
            elif now_s >= self.measurement_end + self.config.support_s:
                self.fail("profile_timeout")
        return self.phase
