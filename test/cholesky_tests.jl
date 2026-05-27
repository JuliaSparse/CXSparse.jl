include("shared.jl")

@testset "CXSparse cs_cholesky" begin
    Random.seed!(0xCC55_F00D)

    @testset "SPD ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (5, 25, 100)

        Mraw = _randmat(Tv, n, n)
        Adense = Mraw * Mraw' + Tv(n) * I
        A = _convert(Adense, Tv, Ti)
        b = _randvec(Tv, n)

        F = cs_cholesky(A)
        @test size(F) == (n, n)
        @test size(F, 1) == n
        @test size(F, 2) == n

        x = F \ b
        @test length(x) == n
        @test norm(A * x - b) < 1e-8 * norm(b)
        @test x ≈ Adense \ b rtol=1e-8

        x_pre = zeros(Tv, n)
        @test ldiv!(x_pre, F, b) === x_pre
        @test x_pre ≈ x
    end

    @testset "rejects non-square ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 3, 5), Tv, Ti)
        @test_throws ErrorException cs_cholesky(A)
    end

    @testset "rejects non-positive-definite ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(Tv[-1 0; 0 -1], Tv, Ti)
        @test_throws ErrorException cs_cholesky(A)
    end

    @testset "rejects zero diagonal ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(Tv[0 0; 0 1], Tv, Ti)
        @test_throws ErrorException cs_cholesky(A)
    end

    @testset "dimension checks ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(Tv[2 0; 0 2], Tv, Ti)
        F = cs_cholesky(A)
        @test_throws DimensionMismatch (F \ Tv[1, 2, 3])
        @test_throws DimensionMismatch ldiv!(zeros(Tv, 3), F, Tv[1, 2])
    end

    @testset "n=1 trivial case ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(reshape(Tv[9.0], 1, 1), Tv, Ti)
        F = cs_cholesky(A)
        @test F \ Tv[18.0] ≈ Tv[2.0]
    end

    @testset "identity matrix ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (3, 10)
        A = _convert(Matrix{Tv}(I, n, n), Tv, Ti)
        b = _randvec(Tv, n)
        F = cs_cholesky(A)
        @test F \ b ≈ b
    end

    @testset "diagonal positive matrix ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        d = Tv <: Complex ? Tv[2+0im, 3+0im, 4+0im, 1+0im] : Tv[2, 3, 4, 1]
        A = _convert(Diagonal(d), Tv, Ti)
        b = _randvec(Tv, 4)
        F = cs_cholesky(A)
        @test F \ b ≈ b ./ d
    end

    @testset "tridiagonal SPD ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        n = 25
        Adense = Tridiagonal(fill(Tv(-1), n - 1), fill(Tv(4), n), fill(Tv(-1), n - 1))
        A = _convert(Matrix(Adense), Tv, Ti)
        b = _randvec(Tv, n)
        F = cs_cholesky(A)
        @test F \ b ≈ Matrix(Adense) \ b rtol=1e-10
    end

    @testset "Hermitian (not just real-symmetric) ($Ti)" for Ti in IDXTYPES
        # Hermitian PD matrix with non-trivial complex off-diagonals.
        Adense = ComplexF64[
            4.0+0.0im   1.0+0.5im   0.0+0.0im
            1.0-0.5im   3.0+0.0im   0.5-0.25im
            0.0+0.0im   0.5+0.25im  2.0+0.0im
        ]
        @assert ishermitian(Adense)
        A = _convert(Adense, ComplexF64, Ti)
        b = ComplexF64[1+0im, 2-1im, -1+0.5im]
        F = cs_cholesky(A)
        x = F \ b
        @test x ≈ Adense \ b rtol=1e-10
        @test norm(A * x - b) < 1e-10 * norm(b)
    end

    @testset "small known-answer ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        # A = [4 2; 2 5]  (SPD: dets are 4, 16) ; b = [6, 7]
        # det(A) = 16, x = [(5*6 - 2*7)/16, (4*7 - 2*6)/16] = [16/16, 16/16] = [1, 1]
        A = _convert(Tv[4 2; 2 5], Tv, Ti)
        b = Tv[6, 7]
        F = cs_cholesky(A)
        @test F \ b ≈ Tv[1, 1]
    end

    @testset "input matrix is not mutated by factorization or solve ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        M = _randmat(Tv, 8, 8)
        A = _convert(M * M' + Tv(8) * I, Tv, Ti)
        snap = _snapshot(A)
        F = cs_cholesky(A)
        @test _unchanged(A, snap)
        _ = F \ _randvec(Tv, 8)
        @test _unchanged(A, snap)
    end

    @testset "rhs `b` is not mutated by solve ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        M = _randmat(Tv, 6, 6)
        A = _convert(M * M' + Tv(6) * I, Tv, Ti)
        b = _randvec(Tv, 6)
        b_orig = copy(b)
        F = cs_cholesky(A)
        _ = F \ b
        @test b == b_orig
        _ = ldiv!(zeros(Tv, 6), F, b)
        @test b == b_orig
    end

    @testset "multiple solves on same factorization ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        M = _randmat(Tv, 10, 10)
        A = _convert(M * M' + Tv(10) * I, Tv, Ti)
        F = cs_cholesky(A)
        Adense = Matrix(A)
        for _ in 1:5
            b = _randvec(Tv, 10)
            x = F \ b
            @test x ≈ Adense \ b rtol=1e-10
        end
    end

    @testset "ldiv! returns the destination vector ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        M = _randmat(Tv, 5, 5)
        A = _convert(M * M' + Tv(5) * I, Tv, Ti)
        F = cs_cholesky(A)
        b = _randvec(Tv, 5)
        x = Vector{Tv}(undef, 5)
        @test ldiv!(x, F, b) === x
    end

    @testset "explicit finalize is safe and idempotent" begin
        M = randn(5, 5)
        A = sparse(M * M' + 5 * I)
        F = cs_cholesky(A)
        _ = F \ randn(5)
        finalize(F)
        finalize(F)
        @test F.S == C_NULL
        @test F.N == C_NULL
    end
end
