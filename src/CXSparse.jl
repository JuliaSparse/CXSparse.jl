module CXSparse

using CXSparse_jll: libcxsparse
using LinearAlgebra: LinearAlgebra, ldiv!
using SparseArrays: SparseArrays, SparseMatrixCSC, getcolptr, rowvals, nonzeros

export cs_qr, cs_lu, cs_cholesky, CSQR, CSLU, CSCholesky

# CXSparse uses 0-based column orderings; pick COLAMD-ish ordering for QR.
# The CSparse `order` argument: 0 = natural, 1 = AMD(A+A'), 2 = AMD(S'S),
# 3 = AMD(A'A). For QR we pass 3 (AMD on A'A).
const CS_ORDER_QR = Int32(3)
# For LU: 2 = AMD(S'S) — Davis's recommended choice for unsymmetric LU.
const CS_ORDER_LU = Int32(2)
# For Cholesky: 1 = AMD(A+A') — symmetric ordering.
const CS_ORDER_CHOL = Int32(1)
# LU pivoting tolerance: 1.0 = strict partial pivoting; smaller values relax it.
const CS_LU_TOL = 1.0

# ---------------------------------------------------------------------------
# C-side struct mirrors
# ---------------------------------------------------------------------------
# `cs_*_sparse` — compressed-column matrix
# `cs_*_symbolic` — symbolic analysis result (`cs_*s`)
# `cs_*_numeric` — numeric factorization result (`cs_*n`)
#
# We mirror them as Julia structs with matching layout. We never read the
# inner pointer-arrays directly; only the C side does. Our job is to set up
# the cs_*_sparse view on a Julia SparseMatrixCSC (with 0-based colptr/rowval
# buffers we own) and to remember the symbolic/numeric pointers so we can
# free them.

for (Tv, Ti, tag) in (
        (:Float64, :Int32, :di),
        (:Float64, :Int64, :dl),
        (:ComplexF64, :Int32, :ci),
        (:ComplexF64, :Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    symbolic_ty = Symbol("cs_$(tag)s")
    numeric_ty = Symbol("cs_$(tag)n")

    @eval begin
        # Layout must match `cs_*_sparse` exactly.
        mutable struct $sparse_ty
            nzmax::$Ti
            m::$Ti
            n::$Ti
            p::Ptr{$Ti}
            i::Ptr{$Ti}
            x::Ptr{$Tv}
            nz::$Ti  # -1 for compressed-col, ≥ 0 for triplet
        end

        # We don't need to introspect the C symbolic/numeric structs; we just
        # hold opaque `Ptr{Cvoid}` to them and pass them back into CXSparse
        # functions. Defining them as `Cvoid` keeps things type-stable
        # without exposing internals we don't need.
    end
end

# ---------------------------------------------------------------------------
# Build a CXSparse `cs_*` view of a SparseMatrixCSC
# ---------------------------------------------------------------------------
# CXSparse expects 0-based indices in `p` (colptr) and `i` (rowval). Julia's
# `SparseMatrixCSC` is 1-based. We allocate new `Int32`/`Int64`-typed buffers
# with the 0-based values and pin them to the wrapper struct so the GC can't
# collect them until the wrapper does.
#
# `nzval` is reused from the Julia matrix directly (CXSparse doesn't mutate
# it during QR / LU factorization of a CSC matrix — it only reads).

struct _CSView{T_sp,Tv,Ti}
    sparse::T_sp        # the cs_di / cs_dl / cs_ci / cs_cl mutable struct
    colptr0::Vector{Ti} # 0-based colptr (owned)
    rowval0::Vector{Ti} # 0-based rowval (owned)
    nzval::Vector{Tv}   # alias to the user's nonzeros (NOT owned)
end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    @eval function _csview(A::SparseMatrixCSC{$Tv,$Ti})
        m, n = size(A)
        cp = collect($Ti, getcolptr(A) .- one($Ti))   # 0-based
        rv = collect($Ti, rowvals(A) .- one($Ti))     # 0-based
        nz = nonzeros(A)
        @assert length(cp) == n + 1
        @assert length(rv) == length(nz)
        sp = $sparse_ty(
            $Ti(length(nz)),                           # nzmax
            $Ti(m),                                    # m
            $Ti(n),                                    # n
            pointer(cp),                               # p
            pointer(rv),                               # i
            pointer(nz),                               # x
            $Ti(-1),                                   # nz = -1 (compressed)
        )
        return _CSView{$sparse_ty,$Tv,$Ti}(sp, cp, rv, nz)
    end
end

# ---------------------------------------------------------------------------
# Symbolic + numeric factorization (QR)
# ---------------------------------------------------------------------------
"""
    CSQR{Tv,Ti}

Symbolic + numeric CXSparse QR factorization of a `SparseMatrixCSC{Tv,Ti}`.

Holds opaque pointers to the C-side `cs_*s` (symbolic) and `cs_*n` (numeric)
structures plus the matrix dimensions and the pinned CSC buffers. The pointers
are freed by a finalizer or explicit `finalize`.

Use `F \\ b` or `ldiv!(x, F, b)` to solve a (possibly rank-deficient or
overdetermined) least-squares system `A x = b`.
"""
mutable struct CSQR{Tv,Ti,T_sp}
    view::_CSView{T_sp,Tv,Ti}
    S::Ptr{Cvoid}       # cs_*s (symbolic)
    N::Ptr{Cvoid}       # cs_*n (numeric)
    m::Int              # rows of A
    n::Int              # cols of A
    m2::Int             # cs_*s->m2 (rows after fictitious-row padding for QR)
end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    sqr_sym = "cs_$(tag)_sqr"
    qr_sym = "cs_$(tag)_qr"
    sfree_sym = "cs_$(tag)_sfree"
    nfree_sym = "cs_$(tag)_nfree"
    happly_sym = "cs_$(tag)_happly"
    usolve_sym = "cs_$(tag)_usolve"
    ipvec_sym = "cs_$(tag)_ipvec"

    @eval begin
        function _cs_sqr_qr(view::_CSView{$sparse_ty,$Tv,$Ti})
            # cs_*s *cs_*_sqr(int*_t order, const cs_* *A, int*_t qr)
            return @ccall libcxsparse.$sqr_sym(
                CS_ORDER_QR::$Ti,
                Ref(view.sparse)::Ref{$sparse_ty},
                $Ti(1)::$Ti,
            )::Ptr{Cvoid}
        end

        function _cs_qr(view::_CSView{$sparse_ty,$Tv,$Ti}, S::Ptr{Cvoid})
            # cs_*n *cs_*_qr(const cs_* *A, const cs_*s *S)
            return @ccall libcxsparse.$qr_sym(
                Ref(view.sparse)::Ref{$sparse_ty},
                S::Ptr{Cvoid},
            )::Ptr{Cvoid}
        end

        # Dispatched on (Tv, Ti) to pick the right cs_*_sfree / cs_*_nfree.
        function _cs_sfree(::Type{$Tv}, ::Type{$Ti}, S::Ptr{Cvoid})
            S == C_NULL && return nothing
            @ccall libcxsparse.$sfree_sym(S::Ptr{Cvoid})::Ptr{Cvoid}
            return nothing
        end
        function _cs_nfree(::Type{$Tv}, ::Type{$Ti}, N::Ptr{Cvoid})
            N == C_NULL && return nothing
            @ccall libcxsparse.$nfree_sym(N::Ptr{Cvoid})::Ptr{Cvoid}
            return nothing
        end

        # Householder reflection applied to a vector: x = (I - beta v v') x,
        # where v is column k of V (= the L slot of the QR numeric struct).
        # NOTE: `beta` is `double` (real) in CXSparse even for complex matrices.
        function _cs_happly!(::Type{$Ti}, V::Ptr{Cvoid}, k::Integer,
                              beta::Cdouble, x::AbstractVector{$Tv})
            @ccall libcxsparse.$happly_sym(
                V::Ptr{Cvoid},
                $Ti(k)::$Ti,
                beta::Cdouble,
                pointer(x)::Ptr{$Tv},
            )::$Ti
        end

        # Solve R x = b in place on b, with R held in the cs_*_sparse `U` slot
        # of the numeric struct.
        function _cs_usolve!(::Type{$Ti}, U::Ptr{Cvoid}, b::AbstractVector{$Tv})
            @ccall libcxsparse.$usolve_sym(
                U::Ptr{Cvoid},
                pointer(b)::Ptr{$Tv},
            )::$Ti
        end

        # Inverse permutation: x[i] = b[p[i]]. With `p == NULL`, just copies.
        function _cs_ipvec!(
                p::Ptr{$Ti},
                b::AbstractVector{$Tv},
                x::AbstractVector{$Tv},
                n::Integer,
            )
            @ccall libcxsparse.$ipvec_sym(
                p::Ptr{$Ti},
                pointer(b)::Ptr{$Tv},
                pointer(x)::Ptr{$Tv},
                $Ti(n)::$Ti,
            )::$Ti
        end
    end
end

# ---------------------------------------------------------------------------
# Read S->m2 / S->q / S->pinv from the cs_*s symbolic struct without
# mirroring its full layout. The struct layout is:
#   int*_t *pinv ; int*_t *q ; int*_t *parent ; int*_t *cp ; int*_t *leftmost ;
#   int*_t m2 ; double lnz ; double unz ;
# so:
#   pinv  is at offset 0          (Ptr{Ti})
#   q     is at offset 1*sizeof(Ptr) (Ptr{Ti})
#   m2    is at offset 5*sizeof(Ptr) (Ti)
# For the numeric struct:
#   cs_* *L ; cs_* *U ; int*_t *pinv ; double *B ;
# so:
#   L     is at offset 0          (Ptr{Cvoid})
#   U     is at offset 1*sizeof(Ptr) (Ptr{Cvoid})
#   B     is at offset 3*sizeof(Ptr) (Ptr{Tv})

@inline _ptr_off(p::Ptr{Cvoid}, k::Integer) = unsafe_load(
    convert(Ptr{Ptr{Cvoid}}, p) + (k - 1) * sizeof(Ptr{Cvoid}),
)
@inline _ti_off(p::Ptr{Cvoid}, k::Integer, ::Type{T}) where {T} = unsafe_load(
    convert(Ptr{T}, p) + (k - 1) * sizeof(Ptr{Cvoid}),  # field is Ti but follows Ptr-aligned slots
)

# Per-Ti accessors for the cs_*s symbolic struct.
for (Tv, Ti) in ((Float64, Int32), (Float64, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval begin
        _cs_S_pinv(::Type{$Tv}, ::Type{$Ti}, S::Ptr{Cvoid}) =
            convert(Ptr{$Ti}, _ptr_off(S, 1))
        _cs_S_q(::Type{$Tv}, ::Type{$Ti}, S::Ptr{Cvoid}) =
            convert(Ptr{$Ti}, _ptr_off(S, 2))
        function _cs_S_m2(::Type{$Tv}, ::Type{$Ti}, S::Ptr{Cvoid})
            # m2 is after 5 pointer-sized slots; size of int*_t may be smaller
            # than a pointer, but CXSparse aligns m2 on a pointer boundary, so
            # we read $Ti at offset 5*sizeof(Ptr).
            return unsafe_load(
                convert(Ptr{$Ti}, S + 5 * sizeof(Ptr{Cvoid})),
            )
        end
        _cs_N_L(::Type{$Tv}, ::Type{$Ti}, N::Ptr{Cvoid}) = _ptr_off(N, 1)
        _cs_N_U(::Type{$Tv}, ::Type{$Ti}, N::Ptr{Cvoid}) = _ptr_off(N, 2)
        # cs_*_numeric.B is `double *` regardless of complex vs real (Householder
        # betas are real even when the matrix is complex).
        function _cs_N_B(::Type{$Tv}, ::Type{$Ti}, N::Ptr{Cvoid})
            return convert(Ptr{Cdouble}, _ptr_off(N, 4))
        end
    end
end

# ---------------------------------------------------------------------------
# Public QR API
# ---------------------------------------------------------------------------
"""
    cs_qr(A::SparseMatrixCSC) -> CSQR

Compute the symbolic and numeric CXSparse QR factorization of `A`. `A` must
be of element type `Float64` or `ComplexF64` and index type `Int32` or `Int64`.
Works for square and overdetermined `A` (m ≥ n); underdetermined matrices
fall back to the transpose-based path internally (see `cs_qrsol` in CXSparse).

Returns a `CSQR` object with a finalizer that frees the C-side memory.
"""
function cs_qr end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    @eval function cs_qr(A::SparseMatrixCSC{$Tv,$Ti})
        m, n = size(A)
        if m < n
            error("CXSparse cs_qr currently requires m ≥ n (overdetermined " *
                  "or square systems). For underdetermined systems consider " *
                  "factoring Aᵀ. Got size $(size(A)).")
        end
        view = _csview(A)
        S = _cs_sqr_qr(view)
        if S == C_NULL
            error("CXSparse cs_*_sqr returned NULL for size $(size(A))")
        end
        N = _cs_qr(view, S)
        if N == C_NULL
            _cs_sfree_qr(S)
            error("CXSparse cs_*_qr returned NULL for size $(size(A))")
        end
        m2 = Int(_cs_S_m2($Tv, $Ti, S))
        F = CSQR{$Tv,$Ti,$sparse_ty}(view, S, N, m, n, m2)
        finalizer(_finalize_qr!, F)
        return F
    end
end

function _finalize_qr!(F::CSQR{Tv,Ti,T_sp}) where {Tv,Ti,T_sp}
    _cs_sfree(Tv, Ti, F.S)
    _cs_nfree(Tv, Ti, F.N)
    F.S = C_NULL
    F.N = C_NULL
    return nothing
end

Base.size(F::CSQR) = (F.m, F.n)
Base.size(F::CSQR, d::Integer) = d == 1 ? F.m : d == 2 ? F.n : 1

"""
    ldiv!(x, F::CSQR, b)

Solve `F \\ b` and store the result in `x`. `b` has length equal to `size(F, 1)`,
`x` has length equal to `size(F, 2)`.
"""
function LinearAlgebra.ldiv!(
        x::AbstractVector{Tv},
        F::CSQR{Tv,Ti,T_sp},
        b::AbstractVector{Tv},
    ) where {Tv,Ti,T_sp}
    m, n = F.m, F.n
    length(b) == m || throw(DimensionMismatch(
        "rhs has length $(length(b)), expected $m"))
    length(x) == n || throw(DimensionMismatch(
        "solution has length $(length(x)), expected $n"))
    # Workspace of size m2 (fictitious-row padded). For square A m2 == m.
    work = Vector{Tv}(undef, max(F.m2, m))
    # x_work[0:m-1] = b[pinv[0:m-1]]
    Spinv = _cs_S_pinv(Tv, Ti, F.S)
    Sq = _cs_S_q(Tv, Ti, F.S)
    L = _cs_N_L(Tv, Ti, F.N)
    U = _cs_N_U(Tv, Ti, F.N)
    Bptr = _cs_N_B(Tv, Ti, F.N)
    _cs_ipvec!(Spinv, b, work, m)
    # Zero-pad m2..m2-1 if any (cs_sqr may add fictitious rows for QR)
    for i in (m + 1):F.m2
        @inbounds work[i] = zero(Tv)
    end
    # Apply Q' to work using the n Householder vectors stored in V (= L).
    # Bptr is `Ptr{Cdouble}` (real betas, see comment on `_cs_N_B`).
    for k in 0:(n - 1)
        beta = unsafe_load(Bptr + k * sizeof(Cdouble))
        _cs_happly!(Ti, L, k, beta, work)
    end
    # Solve R x = work[0:n-1] in place on work
    _cs_usolve!(Ti, U, work)
    # x[q[0:n-1]] = work[0:n-1]
    _cs_ipvec!(Sq, work, x, n)
    return x
end

function Base.:\(F::CSQR{Tv,Ti,T_sp}, b::AbstractVector{Tv}) where {Tv,Ti,T_sp}
    x = Vector{Tv}(undef, F.n)
    return ldiv!(x, F, b)
end

# ---------------------------------------------------------------------------
# LU (exposed for completeness; LinearSolve.jl wires only the QR path)
# ---------------------------------------------------------------------------
"""
    CSLU{Tv,Ti}

Symbolic + numeric CXSparse LU factorization of a square `SparseMatrixCSC{Tv,Ti}`.

For square matrices only. Solve via `F \\ b` or `ldiv!(x, F, b)`.
"""
mutable struct CSLU{Tv,Ti,T_sp}
    view::_CSView{T_sp,Tv,Ti}
    S::Ptr{Cvoid}
    N::Ptr{Cvoid}
    n::Int
end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    sqr_sym = "cs_$(tag)_sqr"
    lu_sym = "cs_$(tag)_lu"
    lsolve_sym = "cs_$(tag)_lsolve"
    usolve_sym = "cs_$(tag)_usolve"
    ipvec_sym = "cs_$(tag)_ipvec"

    @eval begin
        function _cs_sqr_lu(view::_CSView{$sparse_ty,$Tv,$Ti})
            return @ccall libcxsparse.$sqr_sym(
                CS_ORDER_LU::$Ti,
                Ref(view.sparse)::Ref{$sparse_ty},
                $Ti(0)::$Ti,
            )::Ptr{Cvoid}
        end
        function _cs_lu_num(view::_CSView{$sparse_ty,$Tv,$Ti}, S::Ptr{Cvoid})
            return @ccall libcxsparse.$lu_sym(
                Ref(view.sparse)::Ref{$sparse_ty},
                S::Ptr{Cvoid},
                CS_LU_TOL::Cdouble,
            )::Ptr{Cvoid}
        end
        function _cs_lsolve!(::Type{$Ti}, L::Ptr{Cvoid}, b::AbstractVector{$Tv})
            @ccall libcxsparse.$lsolve_sym(
                L::Ptr{Cvoid},
                pointer(b)::Ptr{$Tv},
            )::$Ti
        end
        # For LU, the numeric struct's pinv is at offset 2 (after L, U).
        _cs_N_pinv_lu(::Type{$Tv}, ::Type{$Ti}, N::Ptr{Cvoid}) =
            convert(Ptr{$Ti}, _ptr_off(N, 3))
    end
end

"""
    cs_lu(A::SparseMatrixCSC) -> CSLU

Symbolic + numeric CXSparse LU factorization. `A` must be square.
"""
function cs_lu end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    @eval function cs_lu(A::SparseMatrixCSC{$Tv,$Ti})
        m, n = size(A)
        m == n || error("CXSparse cs_lu requires a square matrix; got $(size(A))")
        view = _csview(A)
        S = _cs_sqr_lu(view)
        S == C_NULL && error("CXSparse cs_*_sqr (LU) returned NULL")
        N = _cs_lu_num(view, S)
        if N == C_NULL
            _cs_sfree($Tv, $Ti, S)
            error("CXSparse cs_*_lu returned NULL (singular?)")
        end
        F = CSLU{$Tv,$Ti,$sparse_ty}(view, S, N, n)
        finalizer(_finalize_lu!, F)
        return F
    end
end

function _finalize_lu!(F::CSLU{Tv,Ti,T_sp}) where {Tv,Ti,T_sp}
    _cs_sfree(Tv, Ti, F.S); _cs_nfree(Tv, Ti, F.N)
    F.S = C_NULL; F.N = C_NULL
    return nothing
end

Base.size(F::CSLU) = (F.n, F.n)
Base.size(F::CSLU, d::Integer) = d == 1 || d == 2 ? F.n : 1

function LinearAlgebra.ldiv!(
        x::AbstractVector{Tv},
        F::CSLU{Tv,Ti,T_sp},
        b::AbstractVector{Tv},
    ) where {Tv,Ti,T_sp}
    n = F.n
    length(b) == n || throw(DimensionMismatch(
        "rhs has length $(length(b)), expected $n"))
    length(x) == n || throw(DimensionMismatch(
        "solution has length $(length(x)), expected $n"))
    work = Vector{Tv}(undef, n)
    Sq = _cs_S_q(Tv, Ti, F.S)
    Npinv = _cs_N_pinv_lu(Tv, Ti, F.N)
    L = _cs_N_L(Tv, Ti, F.N)
    U = _cs_N_U(Tv, Ti, F.N)
    # x_work[0:n-1] = b[pinv[0:n-1]] (row permutation from partial pivoting)
    _cs_ipvec!(Npinv, b, work, n)
    _cs_lsolve!(Ti, L, work)
    _cs_usolve!(Ti, U, work)
    _cs_ipvec!(Sq, work, x, n)
    return x
end

function Base.:\(F::CSLU{Tv,Ti,T_sp}, b::AbstractVector{Tv}) where {Tv,Ti,T_sp}
    x = Vector{Tv}(undef, F.n)
    return ldiv!(x, F, b)
end

# ---------------------------------------------------------------------------
# Cholesky (symmetric/Hermitian positive definite)
# ---------------------------------------------------------------------------
# CXSparse's `cs_*_chol` produces a lower-triangular L with L*L' = P*A*P',
# where P is the AMD(A+A') fill-reducing permutation stored in S->pinv.
# Only the upper triangle of A is read; the matrix is assumed symmetric
# (for Float64) or Hermitian (for ComplexF64).
"""
    CSCholesky{Tv,Ti}

Symbolic + numeric CXSparse Cholesky factorization of a square symmetric
(real) or Hermitian (complex) positive-definite `SparseMatrixCSC{Tv,Ti}`.
Holds opaque pointers to the C-side `cs_*s` symbolic and `cs_*n` numeric
structs; freed by a finalizer or explicit `finalize`.
"""
mutable struct CSCholesky{Tv,Ti,T_sp}
    view::_CSView{T_sp,Tv,Ti}
    S::Ptr{Cvoid}
    N::Ptr{Cvoid}
    n::Int
end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    schol_sym = "cs_$(tag)_schol"
    chol_sym = "cs_$(tag)_chol"
    ltsolve_sym = "cs_$(tag)_ltsolve"
    pvec_sym = "cs_$(tag)_pvec"

    @eval begin
        function _cs_schol(view::_CSView{$sparse_ty,$Tv,$Ti})
            return @ccall libcxsparse.$schol_sym(
                CS_ORDER_CHOL::$Ti,
                Ref(view.sparse)::Ref{$sparse_ty},
            )::Ptr{Cvoid}
        end
        function _cs_chol(view::_CSView{$sparse_ty,$Tv,$Ti}, S::Ptr{Cvoid})
            return @ccall libcxsparse.$chol_sym(
                Ref(view.sparse)::Ref{$sparse_ty},
                S::Ptr{Cvoid},
            )::Ptr{Cvoid}
        end
        function _cs_ltsolve!(::Type{$Ti}, L::Ptr{Cvoid}, b::AbstractVector{$Tv})
            @ccall libcxsparse.$ltsolve_sym(
                L::Ptr{Cvoid},
                pointer(b)::Ptr{$Tv},
            )::$Ti
        end
        function _cs_pvec!(
                p::Ptr{$Ti},
                b::AbstractVector{$Tv},
                x::AbstractVector{$Tv},
                n::Integer,
            )
            @ccall libcxsparse.$pvec_sym(
                p::Ptr{$Ti},
                pointer(b)::Ptr{$Tv},
                pointer(x)::Ptr{$Tv},
                $Ti(n)::$Ti,
            )::$Ti
        end
    end
end

"""
    cs_cholesky(A::SparseMatrixCSC) -> CSCholesky

Compute the symbolic and numeric CXSparse Cholesky factorization of `A`.
`A` must be square and symmetric (real) or Hermitian (complex) positive
definite. Only the upper triangle of `A` is read.

Throws if the factorization fails (typically: `A` is not positive definite,
or numerical breakdown encountered a non-positive pivot).

Solve with `F \\ b` or `ldiv!(x, F, b)`.
"""
function cs_cholesky end

for (Tv, Ti, tag) in (
        (Float64, Int32, :di),
        (Float64, Int64, :dl),
        (ComplexF64, Int32, :ci),
        (ComplexF64, Int64, :cl),
    )
    sparse_ty = Symbol("cs_$(tag)")
    @eval function cs_cholesky(A::SparseMatrixCSC{$Tv,$Ti})
        m, n = size(A)
        m == n || error(
            "CXSparse cs_cholesky requires a square matrix; got $(size(A))")
        view = _csview(A)
        S = _cs_schol(view)
        S == C_NULL && error("CXSparse cs_*_schol returned NULL")
        N = _cs_chol(view, S)
        if N == C_NULL
            _cs_sfree($Tv, $Ti, S)
            error("CXSparse cs_*_chol returned NULL — matrix not positive " *
                  "definite (or not symmetric/Hermitian)?")
        end
        F = CSCholesky{$Tv,$Ti,$sparse_ty}(view, S, N, n)
        finalizer(_finalize_chol!, F)
        return F
    end
end

function _finalize_chol!(F::CSCholesky{Tv,Ti,T_sp}) where {Tv,Ti,T_sp}
    _cs_sfree(Tv, Ti, F.S); _cs_nfree(Tv, Ti, F.N)
    F.S = C_NULL; F.N = C_NULL
    return nothing
end

Base.size(F::CSCholesky) = (F.n, F.n)
Base.size(F::CSCholesky, d::Integer) = d == 1 || d == 2 ? F.n : 1

function LinearAlgebra.ldiv!(
        x::AbstractVector{Tv},
        F::CSCholesky{Tv,Ti,T_sp},
        b::AbstractVector{Tv},
    ) where {Tv,Ti,T_sp}
    n = F.n
    length(b) == n || throw(DimensionMismatch(
        "rhs has length $(length(b)), expected $n"))
    length(x) == n || throw(DimensionMismatch(
        "solution has length $(length(x)), expected $n"))
    work = Vector{Tv}(undef, n)
    Spinv = _cs_S_pinv(Tv, Ti, F.S)
    L = _cs_N_L(Tv, Ti, F.N)
    # work = P*b
    _cs_ipvec!(Spinv, b, work, n)
    # work = L \ work
    _cs_lsolve!(Ti, L, work)
    # work = L' \ work
    _cs_ltsolve!(Ti, L, work)
    # x = P' * work
    _cs_pvec!(Spinv, work, x, n)
    return x
end

function Base.:\(F::CSCholesky{Tv,Ti,T_sp}, b::AbstractVector{Tv}) where {Tv,Ti,T_sp}
    x = Vector{Tv}(undef, F.n)
    return ldiv!(x, F, b)
end

end # module
