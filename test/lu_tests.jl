include("shared.jl")

@testset "CXSparse cs_lu" begin
    Random.seed!(0xCC55_F00D)

    @testset "square well-conditioned ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (5, 25, 100)
        Adense = _randmat(Tv, n, n) + Tv(n) * I
        A = _convert(Adense, Tv, Ti)
        b = _randvec(Tv, n)

        F = cs_lu(A)
        @test size(F) == (n, n)
        x = F \ b
        @test norm(A * x - b) < 1e-8 * norm(b)
        @test x ≈ Adense \ b rtol=1e-8

        x_pre = zeros(Tv, n)
        @test ldiv!(x_pre, F, b) === x_pre
        @test x_pre ≈ x
    end

    @testset "rejects non-square ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 3, 5), Tv, Ti)
        @test_throws ErrorException cs_lu(A)
        A2 = _convert(_randmat(Tv, 5, 3), Tv, Ti)
        @test_throws ErrorException cs_lu(A2)
    end

    @testset "singular matrix is rejected ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        # A rank-deficient square matrix: row 2 == 2 × row 1.
        A = _convert(Tv[1 2; 2 4], Tv, Ti)
        @test_throws ErrorException cs_lu(A)
    end

    @testset "dimension checks ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(Tv[2 0; 0 3], Tv, Ti)
        F = cs_lu(A)
        @test_throws DimensionMismatch (F \ Tv[1, 2, 3])
        @test_throws DimensionMismatch ldiv!(zeros(Tv, 3), F, Tv[1, 2])
        @test_throws DimensionMismatch ldiv!(zeros(Tv, 2), F, Tv[1])
    end

    @testset "n=1 trivial case ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(reshape(Tv[5.0], 1, 1), Tv, Ti)
        F = cs_lu(A)
        x = F \ Tv[10.0]
        @test x ≈ Tv[2.0]
    end

    @testset "identity matrix ($Tv, $Ti, n=$n)" for Tv in ELTYPES,
            Ti in IDXTYPES, n in (3, 10)
        A = _convert(Matrix{Tv}(I, n, n), Tv, Ti)
        b = _randvec(Tv, n)
        F = cs_lu(A)
        @test F \ b ≈ b
    end

    @testset "diagonal matrix ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        d = Tv <: Complex ? Tv[2+0im, -3+1im, 4-2im, 1+0im] : Tv[2, -3, 4, 1]
        A = _convert(Diagonal(d), Tv, Ti)
        b = _randvec(Tv, 4)
        F = cs_lu(A)
        @test F \ b ≈ b ./ d
    end

    @testset "requires pivoting (zero on diagonal) ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        # Without pivoting, LU breaks at A[1,1] = 0. Partial pivoting must swap
        # rows. Exact solution exists: x = [1, 1].
        A = _convert(Tv[0 2; 1 0], Tv, Ti)
        b = Tv[2, 1]
        F = cs_lu(A)
        @test F \ b ≈ Tv[1, 1]
    end

    @testset "upper-triangular ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        # A = [2 1 0; 0 3 1; 0 0 4]; b = [5, 7, 8] ⇒ back-sub: x3=2, x2=5/3, x1=5/3
        A = _convert(Tv[2 1 0; 0 3 1; 0 0 4], Tv, Ti)
        b = Tv[5, 7, 8]
        F = cs_lu(A)
        @test F \ b ≈ Matrix(A) \ b
    end

    @testset "tridiagonal ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        n = 25
        Adense = Tridiagonal(fill(Tv(-1), n - 1), fill(Tv(4), n), fill(Tv(-1), n - 1))
        A = _convert(Matrix(Adense), Tv, Ti)
        b = _randvec(Tv, n)
        F = cs_lu(A)
        @test F \ b ≈ Matrix(Adense) \ b rtol=1e-10
    end

    @testset "input matrix is not mutated by factorization or solve ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 8, 8) + Tv(8) * I, Tv, Ti)
        snap = _snapshot(A)
        F = cs_lu(A)
        @test _unchanged(A, snap)
        _ = F \ _randvec(Tv, 8)
        @test _unchanged(A, snap)
    end

    @testset "rhs `b` is not mutated by solve ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 6, 6) + Tv(6) * I, Tv, Ti)
        b = _randvec(Tv, 6)
        b_orig = copy(b)
        F = cs_lu(A)
        _ = F \ b
        @test b == b_orig
        _ = ldiv!(zeros(Tv, 6), F, b)
        @test b == b_orig
    end

    @testset "multiple solves on same factorization ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 10, 10) + Tv(10) * I, Tv, Ti)
        F = cs_lu(A)
        Adense = Matrix(A)
        for _ in 1:5
            b = _randvec(Tv, 10)
            x = F \ b
            @test x ≈ Adense \ b rtol=1e-10
        end
    end

    @testset "ldiv! returns the destination vector ($Tv, $Ti)" for
            Tv in ELTYPES, Ti in IDXTYPES
        A = _convert(_randmat(Tv, 5, 5) + Tv(5) * I, Tv, Ti)
        F = cs_lu(A)
        b = _randvec(Tv, 5)
        x = Vector{Tv}(undef, 5)
        ret = ldiv!(x, F, b)
        @test ret === x
    end

    @testset "small known-answer ($Tv, $Ti)" for Tv in ELTYPES, Ti in IDXTYPES
        # A = [2 1; 1 3]; b = [3, 4]  ⇒  det=5, x = [(3*3 - 1*4)/5, (2*4 - 1*3)/5] = [1, 1]
        A = _convert(Tv[2 1; 1 3], Tv, Ti)
        b = Tv[3, 4]
        F = cs_lu(A)
        @test F \ b ≈ Tv[1, 1]
    end

    @testset "explicit finalize is safe and idempotent" begin
        A = sparse(randn(5, 5) + 5 * I)
        F = cs_lu(A)
        _ = F \ randn(5)
        finalize(F)
        finalize(F)
        @test F.S == C_NULL
        @test F.N == C_NULL
    end
end
