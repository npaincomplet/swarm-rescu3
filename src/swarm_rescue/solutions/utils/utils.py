import numpy as np

def bresenham(a: np.ndarray, b: np.ndarray):
    r1, c1 = map(int, a)
    r2, c2 = map(int, b)

    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc

    r, c = r1, c1
    while True:
        yield r, c
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc