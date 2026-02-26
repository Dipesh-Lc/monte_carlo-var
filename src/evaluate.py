def basel_traffic_light(n_exceptions_250):
    """
    Basel traffic light for 99% VaR over 250 trading days (commonly used benchmark):
      Green: 0-4 exceptions
      Yellow: 5-9 exceptions
      Red: 10+ exceptions
    Returns: (zone, explanation)
    """
    x = int(n_exceptions_250)

    if x <= 4:
        return "GREEN", "0-4 exceptions (acceptable)"
    elif x <= 9:
        return "YELLOW", "5-9 exceptions (increased scrutiny / multiplier)"
    else:
        return "RED", "10+ exceptions (model likely inadequate)"