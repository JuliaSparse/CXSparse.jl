using CXSparse
using LinearAlgebra
using SparseArrays
using Test
using Random

const ELTYPES = (Float64, ComplexF64)
const IDXTYPES = (Int32, Int64)

# Convert a generic dense/sparse matrix to the (Tv, Ti) variant. CXSparse_jll
# ships only Float64/ComplexF64 × Int32/Int64.
_convert(A::AbstractMatrix, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti} =
    SparseMatrixCSC{Tv,Ti}(sparse(A))
_convert(A::SparseMatrixCSC, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti} =
    SparseMatrixCSC{Tv,Ti}(A)

# Random vector of the right element type.
_randvec(::Type{T}, n::Integer) where {T<:Real} = Vector{T}(randn(n))
_randvec(::Type{T}, n::Integer) where {T<:Complex} =
    Vector{T}(complex.(randn(n), randn(n)))

# Random dense matrix of the right element type.
_randmat(::Type{T}, m::Integer, n::Integer) where {T<:Real} = randn(m, n)
_randmat(::Type{T}, m::Integer, n::Integer) where {T<:Complex} =
    complex.(randn(m, n), randn(m, n))

# Snapshot the storage of a SparseMatrixCSC so we can verify it is not mutated
# by factorization. Returns deep copies of (colptr, rowval, nzval).
_snapshot(A::SparseMatrixCSC) = (copy(A.colptr), copy(A.rowval), copy(A.nzval))

# Compare a snapshot with the current state of A.
function _unchanged(A::SparseMatrixCSC, snap)
    cp, rv, nz = snap
    return A.colptr == cp && A.rowval == rv && A.nzval == nz
end
