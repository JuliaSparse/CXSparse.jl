using CXSparse
using LinearAlgebra
using SparseArrays
using Test
using Random

const ELTYPES = (Float64, ComplexF64)
const IDXTYPES = (Int32, Int64)

# Convert a generic sparse matrix to the (Tv, Ti) variant used by a particular
# CXSparse entry point. CXSparse_jll only ships Float64/ComplexF64 × Int32/Int64.
_convert(A::SparseMatrixCSC, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti} =
    SparseMatrixCSC{Tv,Ti}(A)

@testset "CXSparse cs_qr" begin
    Random.seed!(0xCC55_F00D)

    @testset "well-conditioned square ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (5, 25, 100)

        # Sparse-ish, diagonally dominant → non-singular.
        Adense = (Tv <: Complex ? complex.(randn(n, n), randn(n, n)) : randn(n, n)) +
                 Tv(n) * I
        A = _convert(sparse(Adense), Tv, Ti)
        b = (Tv <: Complex ? complex.(randn(n), randn(n)) : randn(n))
        b = Vector{Tv}(b)

        F = cs_qr(A)
        @test size(F) == (n, n)
        @test size(F, 1) == n
        @test size(F, 2) == n

        x = F \ b
        @test length(x) == n
        @test norm(A * x - b) < 1e-8 * norm(b)

        x_pre = zeros(Tv, n)
        @test ldiv!(x_pre, F, b) === x_pre
        @test x_pre ≈ x
    end

    @testset "overdetermined m > n ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        # 5×3 overdetermined system; check that the residual matches the
        # dense least-squares solution to high precision.
        m, n = 5, 3
        Adense = Tv <: Complex ? complex.(randn(m, n), randn(m, n)) : randn(m, n)
        b_ = Tv <: Complex ? complex.(randn(m), randn(m)) : randn(m)
        b = Vector{Tv}(b_)
        A = _convert(sparse(Adense), Tv, Ti)

        F = cs_qr(A)
        @test size(F) == (m, n)
        x_cs = F \ b
        x_ref = Adense \ b
        @test norm(A * x_cs - b) ≈ norm(Adense * x_ref - b) rtol=1e-8
    end

    @testset "rejects underdetermined m < n" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(sparse([1.0 2.0 3.0; 4.0 5.0 6.0]), Tv, Ti)
        @test_throws ErrorException cs_qr(A)
    end

    @testset "dimension checks" begin
        A = sparse([1.0 0.0; 1.0 1.0])
        F = cs_qr(A)
        @test_throws DimensionMismatch (F \ [1.0, 2.0, 3.0])
        @test_throws DimensionMismatch ldiv!(zeros(3), F, [1.0, 2.0])
        @test_throws DimensionMismatch ldiv!(zeros(2), F, [1.0])
    end

    @testset "rank-deficient: solve does not throw" begin
        # cs_qr is NOT rank-revealing — `x` may contain non-finite entries on
        # truly rank-deficient matrices. We only assert that calling factor +
        # solve does not throw. Use SPQR if you need a rank-revealing result.
        A = sparse([1.0 2.0; 0.5 1.0; 2.0 4.0])  # rank 1
        b = [1.0, 1.0, 1.0]
        F = cs_qr(A)
        x = F \ b
        @test x isa Vector
        @test length(x) == size(F, 2)
    end
end

@testset "CXSparse cs_lu" begin
    Random.seed!(0xCC55_F00D)

    @testset "square well-conditioned ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (5, 25, 100)
        Adense = (Tv <: Complex ? complex.(randn(n, n), randn(n, n)) : randn(n, n)) +
                 Tv(n) * I
        A = _convert(sparse(Adense), Tv, Ti)
        b_ = Tv <: Complex ? complex.(randn(n), randn(n)) : randn(n)
        b = Vector{Tv}(b_)

        F = cs_lu(A)
        @test size(F) == (n, n)
        x = F \ b
        @test norm(A * x - b) < 1e-8 * norm(b)
        x_pre = zeros(Tv, n)
        @test ldiv!(x_pre, F, b) === x_pre
        @test x_pre ≈ x
    end

    @testset "rejects non-square" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(sparse(randn(3, 5)), Tv, Ti)
        @test_throws ErrorException cs_lu(A)
    end
end

@testset "CXSparse cs_cholesky" begin
    Random.seed!(0xCC55_F00D)

    @testset "SPD ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (5, 25, 100)

        # SPD (real) / Hermitian PD (complex) via M*M' + n*I.
        Mraw = Tv <: Complex ? complex.(randn(n, n), randn(n, n)) : randn(n, n)
        Adense = Mraw * Mraw' + Tv(n) * I
        # Adense is dense Float64 / ComplexF64 either way; sparsify and convert.
        A = _convert(sparse(Adense), Tv, Ti)
        b_ = Tv <: Complex ? complex.(randn(n), randn(n)) : randn(n)
        b = Vector{Tv}(b_)

        F = cs_cholesky(A)
        @test size(F) == (n, n)
        @test size(F, 1) == n
        @test size(F, 2) == n

        x = F \ b
        @test length(x) == n
        @test norm(A * x - b) < 1e-8 * norm(b)

        x_pre = zeros(Tv, n)
        @test ldiv!(x_pre, F, b) === x_pre
        @test x_pre ≈ x
    end

    @testset "rejects non-square ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(sparse(randn(3, 5)), Tv, Ti)
        @test_throws ErrorException cs_cholesky(A)
    end

    @testset "rejects non-positive-definite" begin
        # Negative-definite — cs_chol returns NULL on the first non-positive
        # pivot, which we surface as an error.
        A = sparse(Float64[-1.0 0.0; 0.0 -1.0])
        @test_throws ErrorException cs_cholesky(A)
    end

    @testset "dimension checks" begin
        A = sparse(Float64[2.0 0.0; 0.0 2.0])
        F = cs_cholesky(A)
        @test_throws DimensionMismatch (F \ [1.0, 2.0, 3.0])
        @test_throws DimensionMismatch ldiv!(zeros(3), F, [1.0, 2.0])
    end
end

@testset "Finalizer doesn't crash" begin
    # Force GC of factorizations and ensure we don't double-free.
    for _ in 1:50
        A = sparse(randn(10, 10) + 10 * I)
        Asym = Matrix(A) * Matrix(A)' + 10 * I
        Asym = sparse(Asym)
        F = cs_qr(A)
        F2 = cs_lu(A)
        F3 = cs_cholesky(Asym)
        _ = F \ randn(10)
        _ = F2 \ randn(10)
        _ = F3 \ randn(10)
    end
    GC.gc()
    GC.gc()
    @test true
end
