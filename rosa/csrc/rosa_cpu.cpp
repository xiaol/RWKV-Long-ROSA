// ROSA: Rapid Online Suffix Automaton (BlinkDL, RWKV-8 "Heron").
// Fast CPU implementation as a torch extension, parallel over independent rows.
//
// Two ops:
//  1) rosa_tokens(x[B,T] int64)  -> pred[B,T], mlen[B,T], src[B,T]  (int64)
//     y_i = x_{j+1} where j<i maximises m s.t. x_{j-m:j} == x_{i-m:i} (largest j on ties);
//     pred=-1 if no earlier occurrence.  mlen = m+1 = matched suffix length.  src = j+1.
//     Large-vocab version: transitions in a global open-addressing hash table.
//  2) rosa_qkv(q[R,T], k[R,T], v[R,T] uint8, alphabet A<=256) -> out[R,T] int16, mlen[R,T] int32, src[R,T] int32
//     BlinkDL's samx_qkv / rosa_slow_ref semantics: at position i find the largest w such that
//     q[i+1-w:i+1] == k[j:j+w] with j+w <= i (largest j on ties); output v[j+w]; -1 if none.
//     Small-alphabet version: dense transition arrays.
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ----------------------------------------------------------------------------------------------
// Large-vocab suffix automaton with a global hash map for transitions.
// ----------------------------------------------------------------------------------------------
struct HashSAM {
    int n_states = 0;
    std::vector<int> link, len, last_end;       // suffix link, longest length, last end position
    std::vector<std::vector<int>> keys;         // tokens with outgoing transitions (for cloning)
    std::vector<uint64_t> hkeys; std::vector<int> hvals; uint64_t mask;
    static constexpr uint64_t EMPTY = ~0ull;

    explicit HashSAM(int T) {
        int cap = 2 * T + 2;
        link.assign(cap, -1); len.assign(cap, 0); last_end.assign(cap, -1); keys.resize(cap);
        uint64_t hs = 16; while (hs < (uint64_t)(8 * T + 16)) hs <<= 1;
        hkeys.assign(hs, EMPTY); hvals.assign(hs, -1); mask = hs - 1;
        n_states = 1; // root = 0
    }
    static inline uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k; }
    inline int get(int s, int64_t t) const {
        uint64_t key = ((uint64_t)s << 32) | (uint64_t)(uint32_t)t; uint64_t h = mix(key) & mask;
        while (true) { if (hkeys[h] == key) return hvals[h]; if (hkeys[h] == EMPTY) return -1; h = (h + 1) & mask; }
    }
    inline void set(int s, int64_t t, int to, bool is_new) {
        uint64_t key = ((uint64_t)s << 32) | (uint64_t)(uint32_t)t; uint64_t h = mix(key) & mask;
        while (true) {
            if (hkeys[h] == key) { hvals[h] = to; return; }
            if (hkeys[h] == EMPTY) { hkeys[h] = key; hvals[h] = to; if (is_new) keys[s].push_back((int)t); return; }
            h = (h + 1) & mask;
        }
    }
    int new_state() { int s = n_states++; return s; }
};

void rosa_tokens_row(const int64_t* x, int T, int K, int64_t* pred, int64_t* mlen, int64_t* src, int64_t* cnt) {
    // Outputs per position i, for k in [0,K): the k-th longest suffix of x[0..i] that occurred before
    // (walking the suffix-link chain), its successor token, its length, the source position and the
    // number of earlier occurrences of that suffix.  k=0 is BlinkDL's ROSA prediction.
    HashSAM A(T);
    std::vector<int64_t> occ(2 * T + 2, 0);  // number of end positions recorded so far per state
    int last = 0;
    for (int i = 0; i < T; ++i) {
        int64_t t = x[i];
        int cur = A.new_state(); A.len[cur] = A.len[last] + 1; int p = last;
        while (p != -1 && A.get(p, t) == -1) { A.set(p, t, cur, true); p = A.link[p]; }
        if (p == -1) A.link[cur] = 0;
        else {
            int q = A.get(p, t);
            if (A.len[p] + 1 == A.len[q]) A.link[cur] = q;
            else {
                int u = A.new_state(); A.len[u] = A.len[p] + 1; A.link[u] = A.link[q]; A.last_end[u] = A.last_end[q]; occ[u] = occ[q];
                A.keys[u] = A.keys[q];
                for (int tk : A.keys[q]) A.set(u, tk, A.get(q, tk), false);
                while (p != -1 && A.get(p, t) == q) { A.set(p, t, u, false); p = A.link[p]; }
                A.link[q] = A.link[cur] = u;
            }
        }
        last = cur;
        // ---- readout: walk suffix links from cur, collecting up to K states with an earlier occurrence
        int v = cur; int k = 0;
        int64_t* pp = pred + (size_t)i * K; int64_t* mp = mlen + (size_t)i * K; int64_t* sp = src + (size_t)i * K; int64_t* cp = cnt + (size_t)i * K;
        while (v != -1 && k < K) {
            if (A.len[v] > 0 && A.last_end[v] >= 0) { pp[k] = x[A.last_end[v] + 1]; mp[k] = A.len[v]; sp[k] = A.last_end[v] + 1; cp[k] = occ[v]; ++k; }
            v = A.link[v];
        }
        for (; k < K; ++k) { pp[k] = -1; mp[k] = 0; sp[k] = -1; cp[k] = 0; }
        // ---- record end position i on the whole suffix path of cur
        v = cur; while (v != -1 && A.last_end[v] < i) { A.last_end[v] = i; occ[v] += 1; v = A.link[v]; }
    }
}

// ----------------------------------------------------------------------------------------------
// Small-alphabet QKV ROSA (dense transitions).  Automaton built on K; Q walked as a pattern.
// ----------------------------------------------------------------------------------------------
struct DenseSAM {
    int A; int n_states = 0;
    std::vector<int> nxt, link, len, last_end;
    DenseSAM(int T, int A_) : A(A_) {
        int cap = 2 * T + 2; nxt.assign((size_t)cap * A, -1); link.assign(cap, -1); len.assign(cap, 0); last_end.assign(cap, -1); n_states = 1;
    }
    inline int& tr(int s, int c) { return nxt[(size_t)s * A + c]; }
};

void rosa_qkv_row(const uint8_t* q, const uint8_t* k, const uint8_t* v, int T, int A,
                  int16_t* out, int32_t* mlen, int32_t* src) {
    DenseSAM S(T, A);
    int last = 0;      // last state of the K automaton (K[0..i-1])
    int w_state = 0;   // state reached by walking the Q pattern in the K automaton
    int w_len = 0;     // length of the current Q-suffix match inside K[0..i-1]
    for (int i = 0; i < T; ++i) {
        // ---- extend the Q match by q[i] against automaton of K[0..i-1]
        int qc = q[i];
        int p = w_state, x = w_len;
        while (p != -1 && S.tr(p, qc) == -1) { if (x > S.len[p]) x = S.len[p]; p = S.link[p]; x = (p == -1) ? 0 : std::min(x, S.len[p]); }
        // note: after moving to link[p] the matched length is min(x, len[link]) -- handled above via clamps
        if (p == -1) { p = 0; x = 0; } else { p = S.tr(p, qc); x = x + 1; }
        // ---- choose the state that represents exactly the match of length x (walk down links while len[link] >= x)
        int vst = p;
        while (S.link[vst] != -1 && S.len[S.link[vst]] >= x) vst = S.link[vst];
        // ---- find the longest match with a recorded occurrence entirely inside K[0..i-1]
        int rv = vst; int m = x;
        while (rv != -1 && (S.len[rv] <= 0 || S.last_end[rv] < 0)) { rv = S.link[rv]; if (rv != -1) m = std::min(m, S.len[rv]); }
        if (rv != -1) { int e = S.last_end[rv]; int pos = e + 1; out[i] = (int16_t)v[pos]; mlen[i] = m; src[i] = pos; }
        else { out[i] = -1; mlen[i] = 0; src[i] = -1; }
        w_state = p; w_len = x;
        // ---- now add k[i] to the K automaton
        int kc = k[i];
        int cur = S.n_states++; S.len[cur] = S.len[last] + 1; p = last;
        while (p != -1 && S.tr(p, kc) == -1) { S.tr(p, kc) = cur; p = S.link[p]; }
        if (p == -1) S.link[cur] = 0;
        else {
            int qq = S.tr(p, kc);
            if (S.len[p] + 1 == S.len[qq]) S.link[cur] = qq;
            else {
                int u = S.n_states++; S.len[u] = S.len[p] + 1; S.link[u] = S.link[qq]; S.last_end[u] = S.last_end[qq];
                std::memcpy(&S.nxt[(size_t)u * A], &S.nxt[(size_t)qq * A], sizeof(int) * A);
                while (p != -1 && S.tr(p, kc) == qq) { S.tr(p, kc) = u; p = S.link[p]; }
                S.link[qq] = S.link[cur] = u;
                // the Q-walk state may have been the split state qq; if the matched length now belongs to u, move it
                if (w_state == qq && w_len <= S.len[u]) w_state = u;
            }
        }
        last = cur;
        int vv = cur; while (vv != -1 && S.last_end[vv] < i) { S.last_end[vv] = i; vv = S.link[vv]; }
    }
}


// ----------------------------------------------------------------------------------------------
// Streaming ROSA for autoregressive decoding: push one token at a time, read K candidates.
// Memory grows O(n) with the context (automaton states), per-token work is amortised O(1).
// ----------------------------------------------------------------------------------------------
struct RosaStream {
    int cap; int K; HashSAM A; std::vector<int64_t> x, occ; int last = 0; int n = 0;
    std::vector<int64_t> pred, mlen, src, cnt;  // candidates for the *last pushed* position
    RosaStream(int64_t capacity, int64_t K_) : cap((int)capacity), K((int)K_), A((int)capacity), occ(2 * capacity + 2, 0) {
        x.reserve(capacity); pred.assign(K, -1); mlen.assign(K, 0); src.assign(K, -1); cnt.assign(K, 0);
    }
    void push(int64_t t) {
        TORCH_CHECK(n < cap, "RosaStream capacity exceeded");
        x.push_back(t); int i = n++;
        int cur = A.new_state(); A.len[cur] = A.len[last] + 1; int p = last;
        while (p != -1 && A.get(p, t) == -1) { A.set(p, t, cur, true); p = A.link[p]; }
        if (p == -1) A.link[cur] = 0;
        else {
            int q = A.get(p, t);
            if (A.len[p] + 1 == A.len[q]) A.link[cur] = q;
            else {
                int u = A.new_state(); A.len[u] = A.len[p] + 1; A.link[u] = A.link[q]; A.last_end[u] = A.last_end[q]; occ[u] = occ[q];
                A.keys[u] = A.keys[q];
                for (int tk : A.keys[q]) A.set(u, tk, A.get(q, tk), false);
                while (p != -1 && A.get(p, t) == q) { A.set(p, t, u, false); p = A.link[p]; }
                A.link[q] = A.link[cur] = u;
            }
        }
        last = cur;
        int v = cur, k = 0;
        while (v != -1 && k < K) {
            if (A.len[v] > 0 && A.last_end[v] >= 0) { pred[k] = x[A.last_end[v] + 1]; mlen[k] = A.len[v]; src[k] = A.last_end[v] + 1; cnt[k] = occ[v]; ++k; }
            v = A.link[v];
        }
        for (; k < K; ++k) { pred[k] = -1; mlen[k] = 0; src[k] = -1; cnt[k] = 0; }
        v = cur; while (v != -1 && A.last_end[v] < i) { A.last_end[v] = i; occ[v] += 1; v = A.link[v]; }
    }
    std::vector<torch::Tensor> candidates() const {
        auto o = torch::dtype(torch::kInt64);
        return {torch::tensor(pred, o), torch::tensor(mlen, o), torch::tensor(src, o), torch::tensor(cnt, o)};
    }
    int64_t size() const { return n; }
    int64_t states() const { return A.n_states; }
};

} // namespace

std::vector<torch::Tensor> rosa_tokens(torch::Tensor x, int64_t K) {
    TORCH_CHECK(x.dtype() == torch::kInt64 && x.dim() == 2 && x.device().is_cpu(), "x must be int64 [B,T] on CPU");
    TORCH_CHECK(K >= 1 && K <= 16, "K in [1,16]");
    x = x.contiguous();
    int B = x.size(0), T = x.size(1);
    auto opts = torch::dtype(torch::kInt64);
    auto pred = torch::empty({B, T, K}, opts), mlen = torch::empty({B, T, K}, opts), src = torch::empty({B, T, K}, opts), cnt = torch::empty({B, T, K}, opts);
    const int64_t* xp = x.data_ptr<int64_t>();
    int64_t *pp = pred.data_ptr<int64_t>(), *mp = mlen.data_ptr<int64_t>(), *sp = src.data_ptr<int64_t>(), *cp = cnt.data_ptr<int64_t>();
#pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; ++b) rosa_tokens_row(xp + (size_t)b * T, T, (int)K, pp + (size_t)b * T * K, mp + (size_t)b * T * K, sp + (size_t)b * T * K, cp + (size_t)b * T * K);
    return {pred, mlen, src, cnt};
}

std::vector<torch::Tensor> rosa_qkv(torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t alphabet) {
    TORCH_CHECK(q.dtype() == torch::kUInt8 && k.dtype() == torch::kUInt8 && v.dtype() == torch::kUInt8, "q,k,v must be uint8");
    TORCH_CHECK(q.dim() == 2 && q.sizes() == k.sizes() && q.sizes() == v.sizes() && q.device().is_cpu(), "q,k,v must be [R,T] on CPU");
    TORCH_CHECK(alphabet >= 2 && alphabet <= 256, "alphabet in [2,256]");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
    int R = q.size(0), T = q.size(1);
    auto out = torch::empty({R, T}, torch::dtype(torch::kInt16));
    auto mlen = torch::empty({R, T}, torch::dtype(torch::kInt32));
    auto src = torch::empty({R, T}, torch::dtype(torch::kInt32));
    const uint8_t *qp = q.data_ptr<uint8_t>(), *kp = k.data_ptr<uint8_t>(), *vp = v.data_ptr<uint8_t>();
    int16_t* op = out.data_ptr<int16_t>(); int32_t* mp = mlen.data_ptr<int32_t>(); int32_t* sp = src.data_ptr<int32_t>();
#pragma omp parallel for schedule(dynamic, 4)
    for (int r = 0; r < R; ++r) rosa_qkv_row(qp + (size_t)r * T, kp + (size_t)r * T, vp + (size_t)r * T, T, (int)alphabet, op + (size_t)r * T, mp + (size_t)r * T, sp + (size_t)r * T);
    return {out, mlen, src};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RosaStream>(m, "RosaStream")
        .def(py::init<int64_t, int64_t>(), py::arg("capacity"), py::arg("K") = 4)
        .def("push", &RosaStream::push)
        .def("candidates", &RosaStream::candidates)
        .def("size", &RosaStream::size)
        .def("states", &RosaStream::states);
    m.def("rosa_tokens", &rosa_tokens, "ROSA over token ids with K-chain: (pred, mlen, src, cnt) each [B,T,K]");
    m.def("rosa_qkv", &rosa_qkv, "ROSA-QKV over small-alphabet symbol rows: (out, mlen, src)");
}
