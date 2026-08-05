"""Tests for the memory-blocked item-kNN recommender (Phase 2, T6).

Pure numpy/scipy — no Spark/Java. Toy cases are hand-worked to full precision in
comments; the similarity math is asserted against those hand values, truncation
and tie-breaking against hand-picked counts, blocking against a block-size sweep,
and scoring against a dense reference.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.models.item_knn import ItemKNNRecommender, build_similarity


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _csr(dense) -> sp.csr_matrix:
    return sp.csr_matrix(np.asarray(dense, dtype=np.float32))


def _tiny_ds(train_csr: sp.csr_matrix) -> EvalDataset:
    """Minimal EvalDataset for fit/score tests (no Spark, no cache files)."""
    n_users, n_items = train_csr.shape
    return EvalDataset(
        cache_dir=None,
        manifest={},
        item_ids=np.array([str(i) for i in range(n_items)], dtype=object),
        user_ids=np.array([str(u) for u in range(n_users)], dtype=object),
        n_train=np.asarray(train_csr.sum(axis=1)).ravel().astype(np.int32),
        train_csr=train_csr,
        item_category_codes=np.zeros(n_items, dtype=np.int32),
    )


# --------------------------------------------------------------------------- #
# toy fixture: 3 users x 4 items                                              #
# --------------------------------------------------------------------------- #
# B (rows = users, cols = items 0..3):
#   u0: {0, 1, 2}   -> [1, 1, 1, 0]
#   u1: {0, 1}      -> [1, 1, 0, 0]
#   u2: {1, 2, 3}   -> [0, 1, 1, 1]
#
# item counts n_i (column sums):  n0=2, n1=3, n2=2, n3=1
# co-occurrence c_ij = #users with both items (BᵀB, off-diagonal):
#   c01=2 (u0,u1)   c02=1 (u0)     c03=0
#   c12=2 (u0,u2)   c13=1 (u2)     c23=1 (u2)
#
# cosine sim(i,j) = c_ij / sqrt(n_i n_j), diagonal 0:
#   sim(0,1) = 2 / sqrt(2*3) = 2/sqrt(6)  = 0.8164965809
#   sim(0,2) = 1 / sqrt(2*2) = 1/2        = 0.5
#   sim(0,3) = 0
#   sim(1,2) = 2 / sqrt(3*2) = 2/sqrt(6)  = 0.8164965809
#   sim(1,3) = 1 / sqrt(3*1) = 1/sqrt(3)  = 0.5773502692
#   sim(2,3) = 1 / sqrt(2*1) = 1/sqrt(2)  = 0.7071067812
TOY_B = _csr(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
    ]
)

_S6 = 2.0 / np.sqrt(6.0)   # 0.8164965809
_HALF = 0.5
_S3 = 1.0 / np.sqrt(3.0)   # 0.5773502692
_S2 = 1.0 / np.sqrt(2.0)   # 0.7071067812

# S is symmetric here (top_n large => no truncation), diagonal zero.
TOY_S_EXPECTED = np.array(
    [
        [0.0,  _S6,  _HALF, 0.0],
        [_S6,  0.0,  _S6,   _S3],
        [_HALF, _S6, 0.0,   _S2],
        [0.0,  _S3,  _S2,   0.0],
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------- #
# 1. toy cosine, hand-computed                                                #
# --------------------------------------------------------------------------- #
def test_toy_cosine_matches_hand_computation():
    S = build_similarity(TOY_B, top_n=10)  # top_n > 4 => nothing truncates
    assert S.dtype == np.float32
    dense = S.toarray()
    assert np.allclose(dense, TOY_S_EXPECTED, atol=1e-6)
    # diagonal removed
    assert np.all(np.diag(dense) == 0.0)
    # symmetric when untruncated
    assert np.allclose(dense, dense.T, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. truncation to top_n=1: clear winners + one index-broken tie             #
# --------------------------------------------------------------------------- #
# B (6 users x 4 items):
#   u0:{0,1} u1:{0,1} u2:{0,2} u3:{1,3} u4:{2,3} u5:{1,2,3}
# counts: n0=3, n1=4, n2=3, n3=3
# co-occurrence:
#   c01=2 (u0,u1)   c02=1 (u2)     c03=0
#   c12=1 (u5)      c13=2 (u3,u5)  c23=2 (u4,u5)
# cosine sims:
#   sim(0,1)=2/sqrt(3*4)=2/(2*sqrt3)=0.5773502692
#   sim(0,2)=1/sqrt(3*3)=1/3        =0.3333333333
#   sim(1,2)=1/sqrt(4*3)=1/(2*sqrt3)=0.2886751346
#   sim(1,3)=2/sqrt(4*3)=2/(2*sqrt3)=0.5773502692
#   sim(2,3)=2/sqrt(3*3)=2/3        =0.6666666667
#
# columns under top_n=1 (each keeps its single best neighbor row):
#   col0 neighbors: row1=0.57735, row2=0.33333  -> keep row1 (clear winner)
#   col1 neighbors: row0=0.57735, row2=0.28868, row3=0.57735
#                   -> TIE at top between row0 and row3 -> keep row0 (ascending idx)
#   col2 neighbors: row0=0.33333, row1=0.28868, row3=0.66667 -> keep row3 (clear)
#   col3 neighbors: row1=0.57735, row2=0.66667 -> keep row2 (clear)
TRUNC_B = _csr(
    [
        [1, 1, 0, 0],  # u0
        [1, 1, 0, 0],  # u1
        [1, 0, 1, 0],  # u2
        [0, 1, 0, 1],  # u3
        [0, 0, 1, 1],  # u4
        [0, 1, 1, 1],  # u5
    ]
)


def test_truncation_top1_keeps_single_best_neighbor_with_tie_by_index():
    S = build_similarity(TRUNC_B, top_n=1).tocsc()
    # exactly one nonzero per column
    counts = np.diff(S.indptr)
    assert np.array_equal(counts, np.ones(4, dtype=counts.dtype))

    # per-column retained (row, value)
    def kept(col):
        lo, hi = S.indptr[col], S.indptr[col + 1]
        return int(S.indices[lo]), float(S.data[lo])

    r0, v0 = kept(0)
    r1, v1 = kept(1)
    r2, v2 = kept(2)
    r3, v3 = kept(3)

    assert r0 == 1 and np.isclose(v0, 2.0 / (2.0 * np.sqrt(3.0)))
    assert r1 == 0 and np.isclose(v1, 2.0 / (2.0 * np.sqrt(3.0)))  # tie -> lower idx
    assert r2 == 3 and np.isclose(v2, 2.0 / 3.0)
    assert r3 == 2 and np.isclose(v3, 2.0 / 3.0)


def test_tie_is_genuine_and_broken_toward_lower_index():
    # The col-1 tie only "proves" tie-breaking if row0 and row3 truly carry the
    # same value; assert that on the untruncated matrix before trusting top_n=1.
    full = build_similarity(TRUNC_B, top_n=10).toarray()
    assert full[0, 1] == full[3, 1]  # exact float equality => genuine tie
    assert full[0, 1] > full[2, 1]  # and it is the column maximum


# --------------------------------------------------------------------------- #
# 3. shrinkage on the toy, hand-computed                                      #
# --------------------------------------------------------------------------- #
# sim(i,j) = c_ij / (sqrt(n_i n_j) + 1), same toy counts as fixture #1:
#   sim(0,1) = 2/(sqrt6+1)  sim(0,2) = 1/(2+1)      sim(0,3)=0
#   sim(1,2) = 2/(sqrt6+1)  sim(1,3) = 1/(sqrt3+1)
#   sim(2,3) = 1/(sqrt2+1)
def test_shrinkage_matches_hand_computation():
    lam = 1.0
    S = build_similarity(TOY_B, top_n=10, shrinkage=lam).toarray()
    s01 = 2.0 / (np.sqrt(6.0) + lam)
    s02 = 1.0 / (2.0 + lam)
    s12 = 2.0 / (np.sqrt(6.0) + lam)
    s13 = 1.0 / (np.sqrt(3.0) + lam)
    s23 = 1.0 / (np.sqrt(2.0) + lam)
    expected = np.array(
        [
            [0.0, s01, s02, 0.0],
            [s01, 0.0, s12, s13],
            [s02, s12, 0.0, s23],
            [0.0, s13, s23, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(S, expected, atol=1e-6)
    assert np.all(np.diag(S) == 0.0)


# --------------------------------------------------------------------------- #
# 4. blocking invariance                                                      #
# --------------------------------------------------------------------------- #
def test_blocking_invariance_block_size_does_not_change_result():
    rng = np.random.default_rng(20260805)
    dense = (rng.random((60, 50)) < 0.15).astype(np.float32)
    B = sp.csr_matrix(dense)

    S_small = build_similarity(B, top_n=8, shrinkage=0.5, block_size=7).tocsr()
    S_big = build_similarity(B, top_n=8, shrinkage=0.5, block_size=50).tocsr()
    S_small.sort_indices()
    S_big.sort_indices()

    # exact byte-identical CSR contents regardless of block boundaries
    assert np.array_equal(S_small.indptr, S_big.indptr)
    assert np.array_equal(S_small.indices, S_big.indices)
    assert np.array_equal(S_small.data, S_big.data)


# --------------------------------------------------------------------------- #
# 5. score_batch == user history rows summed over S columns                   #
# --------------------------------------------------------------------------- #
def test_score_batch_equals_dense_reference_on_toy():
    ds = _tiny_ds(TOY_B)
    model = ItemKNNRecommender(top_n=10)
    model.fit(ds)

    S_dense = TOY_S_EXPECTED  # hand-computed reference
    B_dense = TOY_B.toarray().astype(np.float64)
    expected = B_dense @ S_dense  # (3 users x 4 items)

    scores = model.score_batch(np.array([0, 1, 2]))
    assert scores.shape == (3, 4)
    assert scores.dtype == np.float32
    assert np.allclose(scores, expected, atol=1e-6)

    # spot-check user0 by hand: history {0,1,2}
    #   score_0 = S[0,0]+S[1,0]+S[2,0] = 0 + 0.8164966 + 0.5      = 1.3164966
    #   score_1 = S[0,1]+S[1,1]+S[2,1] = 0.8164966 + 0 + 0.8164966= 1.6329932
    #   score_2 = S[0,2]+S[1,2]+S[2,2] = 0.5 + 0.8164966 + 0      = 1.3164966
    #   score_3 = S[0,3]+S[1,3]+S[2,3] = 0 + 0.5773503 + 0.7071068= 1.2844570
    assert np.allclose(
        scores[0], [1.3164966, 1.6329932, 1.3164966, 1.2844570], atol=1e-6
    )


def test_empty_history_user_scores_all_zero():
    # 3 items; user 2 has no TRAIN interactions.
    B = _csr(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ]
    )
    ds = _tiny_ds(B)
    model = ItemKNNRecommender(top_n=5).fit(ds)
    scores = model.score_batch(np.array([2]))
    assert scores.shape == (1, 3)
    assert np.all(scores == 0.0)


def test_full_catalog_scoring_scores_every_item():
    ds = _tiny_ds(TOY_B)
    model = ItemKNNRecommender(top_n=1).fit(ds)  # aggressive truncation
    scores = model.score_batch(np.array([0]))
    # still (B, I): every catalog item receives a score, none sampled away.
    assert scores.shape == (1, 4)
    assert scores.dtype == np.float32


# --------------------------------------------------------------------------- #
# 6. scale sanity (fast)                                                       #
# --------------------------------------------------------------------------- #
def test_scale_sanity_nnz_per_column_and_dtype():
    rng = np.random.default_rng(20260805)
    n_users, n_items, nnz = 5000, 2000, 30000
    ui = rng.integers(0, n_users, size=nnz)
    ii = rng.integers(0, n_items, size=nnz)
    B = sp.coo_matrix(
        (np.ones(nnz, dtype=np.float32), (ui, ii)),
        shape=(n_users, n_items),
    ).tocsr()
    B.data[:] = 1.0  # binary
    B.eliminate_zeros()

    top_n = 50
    t0 = time.perf_counter()
    S = build_similarity(B, top_n=top_n, block_size=512)
    elapsed = time.perf_counter() - t0

    assert S.dtype == np.float32
    assert S.shape == (n_items, n_items)
    per_col = np.diff(S.tocsc().indptr)
    assert per_col.max() <= top_n
    # diagonal removed
    assert S.diagonal().sum() == 0.0
    # cheap enough to be a unit test
    assert elapsed < 30.0
    print(f"\n[scale-sanity] build_similarity 5000x2000 nnz~30k "
          f"top_n=50 wall={elapsed:.3f}s")
