include("shared.jl")

@testset "CXSparse cs_qr" begin
    Random.seed!(0xCC55_F00D)

    @testset "well-conditioned square ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (5, 25, 100)

        Adense = _randmat(Tv, n, n) + Tv(n) * I
        A = _convert(Adense, Tv, Ti)
        b = _randvec(Tv, n)

        F = cs_qr(A)
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

    @testset "overdetermined m > n: matches dense LS ($Tv, $Ti, m=$m, n=$n)" for
            Tv in ELTYPES, Ti in IDXTYPES, (m, n) in ((5, 3), (10, 4), (50, 10))

        Adense = _randmat(Tv, m, n)
        A = _convert(Adense, Tv, Ti)
        b = _randvec(Tv, m)

        F = cs_qr(A)
        @test size(F) == (m, n)
        x_cs = F \ b
        x_ref = Adense \ b
        # Coefficient-wise match (not just residual norm) — both algorithms
        # should arrive at the same least-squares minimizer.
        @test x_cs ≈ x_ref rtol=1e-8
        @test norm(A * x_cs - b) ≈ norm(Adense * x_ref - b) rtol=1e-8
    end

    @testset "rejects underdetermined m < n ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert([1.0 2.0 3.0; 4.0 5.0 6.0], Tv, Ti)
        @test_throws ErrorException cs_qr(A)
    end

    @testset "dimension checks ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(Tv[1 0; 1 1], Tv, Ti)
        F = cs_qr(A)
        @test_throws DimensionMismatch (F \ Tv[1, 2, 3])
        @test_throws DimensionMismatch ldiv!(zeros(Tv, 3), F, Tv[1, 2])
        @test_throws DimensionMismatch ldiv!(zeros(Tv, 2), F, Tv[1])
    end

    @testset "rank-deficient: solve does not throw" begin
        # cs_qr is NOT rank-revealing — entries may be non-finite. Assert only
        # that the call sequence runs to completion without error.
        A = sparse([1.0 2.0; 0.5 1.0; 2.0 4.0])  # rank 1
        b = [1.0, 1.0, 1.0]
        F = cs_qr(A)
        x = F \ b
        @test x isa Vector
        @test length(x) == size(F, 2)
    end

    @testset "n=1 trivial case ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(reshape(Tv[3.0], 1, 1), Tv, Ti)
        F = cs_qr(A)
        x = F \ Tv[6.0]
        @test x ≈ Tv[2.0]
    end

    @testset "identity matrix ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (3, 10)
        A = _convert(Matrix{Tv}(I, n, n), Tv, Ti)
        b = _randvec(Tv, n)
        F = cs_qr(A)
        x = F \ b
        @test x ≈ b
    end

    @testset "diagonal matrix ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        d = Tv <: Complex ? Tv[2+0im, -3+1im, 4-2im, 1+0im] : Tv[2, -3, 4, 1]
        A = _convert(Diagonal(d), Tv, Ti)
        b = _randvec(Tv, 4)
        F = cs_qr(A)
        x = F \ b
        @test x ≈ b ./ d
    end

    @testset "tridiagonal SPD-like ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        n = 20
        Adense = Tridiagonal(fill(Tv(-1), n - 1), fill(Tv(4), n), fill(Tv(-1), n - 1))
        A = _convert(Matrix(Adense), Tv, Ti)
        b = _randvec(Tv, n)
        F = cs_qr(A)
        x = F \ b
        @test x ≈ Matrix(Adense) \ b rtol=1e-10
    end

    @testset "input matrix is not mutated by factorization or solve ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 8, 8) + Tv(8) * I, Tv, Ti)
        snap = _snapshot(A)
        F = cs_qr(A)
        @test _unchanged(A, snap)
        b = _randvec(Tv, 8)
        _ = F \ b
        @test _unchanged(A, snap)
    end

    @testset "rhs `b` is not mutated by solve ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 6, 6) + Tv(6) * I, Tv, Ti)
        b = _randvec(Tv, 6)
        b_orig = copy(b)
        F = cs_qr(A)
        _ = F \ b
        @test b == b_orig
        _ = ldiv!(zeros(Tv, 6), F, b)
        @test b == b_orig
    end

    @testset "multiple solves on same factorization ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 10, 10) + Tv(10) * I, Tv, Ti)
        F = cs_qr(A)
        Adense = Matrix(A)
        for _ in 1:5
            b = _randvec(Tv, 10)
            x = F \ b
            @test x ≈ Adense \ b rtol=1e-10
        end
    end

    @testset "small known-answer matrix ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        # A upper-triangular: back-substitution by hand.
        # [2 1; 0 3] x = [4; 6]  ⇒  x2 = 2, x1 = (4 - 1*2)/2 = 1
        A = _convert(Tv[2 1; 0 3], Tv, Ti)
        b = Tv[4, 6]
        F = cs_qr(A)
        @test F \ b ≈ Tv[1.0, 2.0]
    end

    @testset "ldiv! returns the destination vector ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 5, 5) + Tv(5) * I, Tv, Ti)
        F = cs_qr(A)
        b = _randvec(Tv, 5)
        x = Vector{Tv}(undef, 5)
        ret = ldiv!(x, F, b)
        @test ret === x
    end

    @testset "explicit finalize is safe and idempotent" begin
        A = sparse(randn(5, 5) + 5 * I)
        F = cs_qr(A)
        b = randn(5)
        _ = F \ b
        finalize(F)
        # A second finalize call must not crash (pointers should be NULL'd).
        finalize(F)
        @test F.S == C_NULL
        @test F.N == C_NULL
    end
end
