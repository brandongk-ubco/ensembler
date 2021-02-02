def RoundUp(x, mul):
    return ((x + mul - 1) & (-mul))
