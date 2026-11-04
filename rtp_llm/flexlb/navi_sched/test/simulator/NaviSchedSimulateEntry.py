import sys


def main():
    if len(sys.argv) < 2:
        raise ValueError("simulate role is required")
    role = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if role == "master":
        from NaviSchedSimulateMaster import main as masterMain

        return masterMain()
    if role in ("agent", "client"):
        from navi_sched_simulate import main as simulateMain

        return simulateMain(role)
    raise ValueError(f"unknown simulate role: {role}")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except (RuntimeError, ValueError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
