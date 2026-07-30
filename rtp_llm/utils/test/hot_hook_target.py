def add(a, b=0):
    return a + b


def raises():
    raise ValueError("boom")


def line_target(value):
    tmp = value + 1
    marker = tmp * 2
    return marker + 3


def other_line_target(value):
    tmp = value + 10
    return tmp


class Sample:
    def method(self, value):
        return value + 1

    @staticmethod
    def static(value):
        return value * 2
