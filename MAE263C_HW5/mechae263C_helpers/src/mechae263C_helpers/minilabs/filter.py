class ExponentialFilter:
    def __init__(self, smoothing_coeff: float, num_warmup_time_steps: int = 0):
        self.smoothing_coeff = max(min(float(smoothing_coeff), 1.0), 0.0)
        self.prev_output = 0.0
        self.is_first_run = True
        self.num_warmup_time_steps = max(round(num_warmup_time_steps), 0)
        self.warmup_time_step_counter = 0

    def __call__(self, x: float) -> float:
        if self.is_first_run:
            self.is_first_run = False
            self.prev_output = x
        else:
            self.prev_output = (
                1 - self.smoothing_coeff
            ) * x + self.smoothing_coeff * self.prev_output

        if self.warmup_time_step_counter <= self.num_warmup_time_steps:
            self.warmup_time_step_counter += 1
            return 0.0
        else:
            return self.prev_output
