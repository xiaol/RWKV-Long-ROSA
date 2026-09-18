"""Reference (slow, pure-Python) ROSA algorithms, transcribed from BlinkDL's RWKV-LM/RWKV-v8.

These exist only to test the fast C++ implementation in ``rosa/csrc``.
"""
from __future__ import annotations


def rosa_tokens_ref(x: list[int]) -> list[int]:
    """BlinkDL's ROSA(x): y_i = x_{j+1} for the longest earlier-occurring suffix, -1 if none.

    Verbatim algorithm from RWKV-v8/251014_rosa_1bit_layer.py (Rapid Online Suffix Automaton).
    """
    n = len(x); y = [-1] * n; s = 2 * n + 1; b = [None] * s; c = [-1] * s; d = [0] * s; e = [-1] * s; b[0] = {}; g = 0; z = 1
    for i, t in enumerate(x):
        r = z; z += 1; b[r] = {}; d[r] = d[g] + 1; p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r; p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z; z += 1; b[u] = b[q].copy(); d[u] = d[p] + 1; c[u] = c[q]; e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u; p = c[p]
                c[q] = c[r] = u
        v = g = r; a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]; break
            v = c[v]
        y[i] = a; v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return y


def rosa_tokens_brute(x: list[int]) -> tuple[list[int], list[int], list[int]]:
    """O(n^3) brute force with the exact tie-breaking rule: max m, then largest j.

    Returns (pred, matched_len, src) where src = j+1 (position of the predicted token).
    """
    n = len(x); pred = [-1] * n; mlen = [0] * n; src = [-1] * n
    for i in range(n):
        best = (-1, -1)  # (m, j)
        for j in range(i - 1, -1, -1):
            m = 0
            while j - m >= 0 and x[j - m] == x[i - m]:
                m += 1
            if m > 0 and m > best[0]:
                best = (m, j)
        if best[0] > 0:
            m, j = best
            pred[i] = x[j + 1]; mlen[i] = m; src[i] = j + 1
    return pred, mlen, src


def rosa_qkv_brute(q: list[int], k: list[int], v: list[int]) -> tuple[list[int], list[int]]:
    """BlinkDL's rosa_slow_ref from RWKV-v7/rwkv_v8_rc00_demo.py (ROSA-4bit), unchanged except
    that unmatched positions return -1 instead of 0 so the two cases are distinguishable."""
    n = len(q); idx = [-1] * n; ln = [0] * n
    for i in range(n):
        found = False
        for w in range(i + 1, 0, -1):
            t = q[i + 1 - w: i + 1]
            for j in range(i - w, -1, -1):
                if k[j: j + w] == t:
                    idx[i] = v[j + w]; ln[i] = w; found = True
                    break
            if found:
                break
    return idx, ln
