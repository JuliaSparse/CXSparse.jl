# CXSparse.jl

Julia wrapper around the [CXSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CXSparse) library from SuiteSparse — the lightweight, textbook-style sparse direct solvers from Tim Davis's *Direct Methods for Sparse Linear Systems* (CSparse), extended to support `ComplexF64` values and both 32- and 64-bit indices.

CXSparse is the QR / LU / Cholesky counterpart to **KLU's design philosophy**: a small symbolic phase, very little per-call overhead, well-suited to small-to-medium sparse problems (n up to a few thousand). It complements the multifrontal solvers (UMFPACK, SPQR, CHOLMOD) which pay a heavier symbolic tax in exchange for BLAS-3 speedups on larger fronts.

This package wraps the three direct factorizations CXSparse provides — `cs_qr`, `cs_lu`, and `cs_chol` — and exposes a Julian `\` / `ldiv!` API on top.

## Status

- [x] `cs_qr` — QR factorization (square or overdetermined `m ≥ n`)
- [x] `cs_lu` — LU factorization (square)
- [x] `cs_cholesky` — Cholesky factorization (square symmetric/Hermitian positive definite)
- [x] All four CXSparse element/index combinations: `Float64` / `ComplexF64` × `Int32` / `Int64`
- [x] `LinearAlgebra.ldiv!` and `\` on every factorization type
- [x] Finalizer-managed C-side memory (no manual `cs_*_sfree` / `cs_*_nfree` calls)

## Quick start

```julia
using CXSparse, SparseArrays, LinearAlgebra

A = sparse([2.0  1.0  0.0;
            1.0  3.0  1.0;
            0.0  1.0  2.0])
b = [1.0, 2.0, 3.0]

# QR factorization (square or overdetermined m ≥ n):
F = cs_qr(A)
x = F \ b
# or in-place into a preallocated buffer:
x_pre = zeros(3)
ldiv!(x_pre, F, b)

# LU factorization (square):
G = cs_lu(A)
y = G \ b

# Cholesky factorization (square symmetric/Hermitian positive definite —
# only the upper triangle of A is read):
H = cs_cholesky(A)
z = H \ b
```

The factorization objects (`CSQR`, `CSLU`, `CSCholesky`) own all C-side memory and free it via a finalizer; you can also call `finalize(F)` explicitly to release it eagerly.

## Supported element/index combinations

| element type   | index type | CXSparse name |
|----------------|------------|---------------|
| `Float64`      | `Int32`    | `cs_di_*`     |
| `Float64`      | `Int64`    | `cs_dl_*`     |
| `ComplexF64`   | `Int32`    | `cs_ci_*`     |
| `ComplexF64`   | `Int64`    | `cs_cl_*`     |

CXSparse does **not** ship `Float32` variants — `cs_si_*` / `cs_sl_*` don't exist upstream. For complex inputs, `cs_cholesky` computes the Hermitian factorization (`L * L' = A`, where `A = A'`).

## When to use this vs SPQR / UMFPACK / CHOLMOD

CXSparse's `cs_qr`, `cs_lu`, and `cs_chol` are textbook implementations: simple, low symbolic-phase overhead, no BLAS-3. They sit alongside KLU in the "small/medium fast path" niche:

| | KLU | UMFPACK | SPQR | CHOLMOD | CXSparse `cs_qr` / `cs_lu` / `cs_cholesky` |
|---|---|---|---|---|---|
| factorization | LU | LU | QR | Cholesky | LU / QR / Cholesky |
| algorithm | BTF + Gilbert–Peierls partial-pivot LU | multifrontal BLAS-3 LU | multifrontal BLAS-3 QR | supernodal BLAS-3 Cholesky | textbook Householder QR / Gilbert–Peierls LU / up-looking Cholesky |
| sweet spot | n ≲ few thousand | n ≳ 1k | n ≳ 1k, least-squares | n ≳ 1k SPD | n ≲ few thousand |
| pivoting | partial | row + column | sparsity-only (rank-revealing if `tol ≥ 0`) | symmetric (no numerical pivoting) | partial (LU), sparsity-only (QR), symmetric (Cholesky) |
| BLAS-3 | no | yes | yes | yes | no |

On 199×199 sparse matrices we observed `cs_qr` running in ~325 µs vs SPQR's ~570 µs (1.7× faster). For rank-deficient inputs, however, `cs_qr` is **not rank-revealing** — `x` may contain non-finite entries from the back-solve dividing through near-zero `R` diagonals. If you need a clean solution on rank-deficient systems, fall back to `qr(A)` (SPQR) or LAPACK's column-pivoted QR on `Matrix(A)`.

`cs_cholesky` returns an error if the matrix is not positive definite (CXSparse aborts the factorization as soon as it hits a non-positive pivot). For semi-definite or indefinite symmetric systems, use `LDLFactorizations.jl` or fall back to `cs_lu`.

## Internals

CXSparse uses 0-based indexing. The wrapper allocates 0-based `colptr` / `rowval` buffers when constructing the `cs_*_sparse` view; `nzval` is shared with the user's `SparseMatrixCSC` directly (CXSparse doesn't mutate it during factorization of a CSC matrix).

Symbolic and numeric C-side structs (`cs_*_symbolic`, `cs_*_numeric`) are owned by the Julia factorization object and freed by a finalizer. The struct internals are accessed via raw pointer offsets matching the layout in `cs.h`; we never mirror the symbolic/numeric structs in full.

## License

The wrapper is MIT-licensed. CXSparse itself is dual-licensed (LGPL-2.1+ / GPL-2.0+) — see the upstream [SuiteSparse repository](https://github.com/DrTimothyAldenDavis/SuiteSparse).
