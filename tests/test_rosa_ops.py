import random, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rosa.ops import rosa_tokens, rosa_qkv_symbols
from rosa.reference import rosa_tokens_ref, rosa_tokens_brute, rosa_qkv_brute


def test_tokens_matches_blinkdl_and_brute():
    rng = random.Random(0)
    for trial in range(300):
        T = rng.randint(1, 60); V = rng.choice([2, 3, 5, 50, 65536])
        x = [rng.randrange(V) for _ in range(T)]
        if rng.random() < 0.3:  # inject repeats to exercise long matches
            L = rng.randint(1, T); x = (x[:L] * (T // L + 1))[:T]
        pred, mlen, src, cnt = rosa_tokens(torch.tensor([x]), K=3)
        pred, mlen, src = pred[..., 0], mlen[..., 0], src[..., 0]
        ref = rosa_tokens_ref(x)
        bp, bm, bs = rosa_tokens_brute(x)
        assert pred[0].tolist() == ref, (x, pred[0].tolist(), ref)
        assert pred[0].tolist() == bp
        assert mlen[0].tolist() == bm, (x, mlen[0].tolist(), bm)
        assert src[0].tolist() == bs, (x, src[0].tolist(), bs)


def test_qkv_matches_brute():
    rng = random.Random(1)
    for trial in range(400):
        T = rng.randint(1, 48); A = rng.choice([2, 2, 4, 16])
        q = [rng.randrange(A) for _ in range(T)]; k = [rng.randrange(A) for _ in range(T)]; v = [rng.randrange(A) for _ in range(T)]
        if rng.random() < 0.4:  # make q copy k with a lag so long matches happen
            lag = rng.randint(1, max(1, T // 2)); q = k[-lag:] + k[:-lag] if lag < T else q
        out, mlen, src = rosa_qkv_symbols(torch.tensor([q], dtype=torch.uint8), torch.tensor([k], dtype=torch.uint8), torch.tensor([v], dtype=torch.uint8), A)
        bi, bl = rosa_qkv_brute(q, k, v)
        assert out[0].tolist() == bi, (q, k, v, out[0].tolist(), bi)
        assert mlen[0].tolist() == bl, (q, k, v, mlen[0].tolist(), bl)


def test_chain_and_counts():
    # brute-force check of the K-chain semantics and occurrence counts
    rng = random.Random(2)
    for trial in range(150):
        T = rng.randint(1, 40); V = rng.choice([2, 3, 4])
        x = [rng.randrange(V) for _ in range(T)]
        K = 4
        pred, mlen, src, cnt = rosa_tokens(torch.tensor([x]), K=K)
        for i in range(T):
            # all suffix lengths m (1..i) of x[:i+1] that occurred earlier, longest first; for each, latest j and count
            cands = []
            for m in range(i + 1, 0, -1):
                suf = x[i + 1 - m: i + 1]
                occ = [j for j in range(m - 1, i) if x[j + 1 - m: j + 1] == suf]
                if occ:
                    cands.append((m, occ[-1] + 1, len(occ)))
            # distinct automaton states: consecutive m with identical occurrence sets collapse; the state keeps the longest m
            # -> emulate by keeping m only if its count differs from the next longer m or it is the longest
            states = []
            for idx_c, (m, s_, c) in enumerate(cands):
                if idx_c == 0 or cands[idx_c - 1][2] != c:
                    states.append((m, s_, c))
            got = [(int(mlen[0, i, k]), int(src[0, i, k]), int(cnt[0, i, k])) for k in range(K) if int(mlen[0, i, k]) > 0]
            assert got == states[:K], (x, i, got, states[:K])
            for k in range(len(got)):
                assert int(pred[0, i, k]) == x[got[k][1]]


def test_speed():
    x = torch.randint(0, 65536, (8, 16384))
    t = time.time(); rosa_tokens(x, K=4); dt = time.time() - t
    print(f"rosa_tokens 8x16384 vocab 65536: {dt*1000:.1f} ms")
    q = torch.randint(0, 2, (1024, 4096), dtype=torch.uint8)
    t = time.time(); rosa_qkv_symbols(q, q, q, 2); dt = time.time() - t
    print(f"rosa_qkv 1024 rows x 4096 (1-bit): {dt*1000:.1f} ms")


if __name__ == "__main__":
    test_tokens_matches_blinkdl_and_brute(); print("tokens OK")
    test_qkv_matches_brute(); print("qkv OK")
    test_chain_and_counts(); print("chain OK")
    test_speed()
