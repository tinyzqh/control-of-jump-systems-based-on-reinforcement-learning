class PID(object):
    def __init__(self, kp, ki, kd, args):
        super(PID).__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = args.dt

        self.cumulative_error = 0.0
        self.last_error = None

    def update(self, error):
        """
        compute the out of fixed PID parameter.
        :param error:
        :return:
        """
        p = self.kp * error
        i = self.ki * self.cumulative_error * self.dt
        if self.last_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self.last_error) / self.dt

        self.cumulative_error += error
        self.last_error = error
        return p + i + d

    def update_with_parameter(self, error, kp, ki, kd):
        """
        compute the out of the learned PID parameter.
        :param error:
        :param kp:
        :param ki:
        :param kd:
        :return:
        """
        p = kp * error
        i = ki * self.cumulative_error * self.dt
        if self.last_error is None:
            d = 0.0
        else:
            d = kd * (error - self.last_error) / self.dt

        self.cumulative_error += error
        self.last_error = error
        return p + i + d

