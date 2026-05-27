include("shared.jl")

@testset "Cross-factorization consistency" begin
    Random.seed!(0xCC55_F00D)

    # On a well-conditioned square non-singular matrix, cs_qr and cs_lu must
    # produce solutions that match each other (and the dense reference) to
    # high precision. On an SPD matrix, cs_cholesky must also agree.

    @testset "QR vs LU on square non-singular ($Tv, $Ti, n=$n)" for
            Tv in ELTYPES, Ti in IDXTYPES, n in (8, 30)
        Adense = _randmat(Tv, n, n) + Tv(n) * I
        A = _convert(Adense, Tv, Ti)
        b = _randvec(Tv, n)

        x_qr = cs_qr(A) \ b
        x_lu = cs_lu(A) \ b
        x_ref = Adense \ b
        @test x_qr ≈ x_lu rtol=1e-9
        @test x_qr ≈ x_ref rtol=1e-9
        @test x_lu ≈ x_ref rtol=1e-9
    end

    @testset "QR vs LU vs Cholesky on SPD ($Tv, $Ti, n=$n)" for
            Tv in ELTYPES, Ti in IDXTYPES, n in (8, 30)
        M = _randmat(Tv, n, n)
        Adense = M * M' + Tv(n) * I
        A = _convert(Adense, Tv, Ti)
        b = _randvec(Tv, n)

        x_qr = cs_qr(A) \ b
        x_lu = cs_lu(A) \ b
        x_chol = cs_cholesky(A) \ b
        x_ref = Adense \ b
        @test x_qr ≈ x_lu rtol=1e-9
        @test x_lu ≈ x_chol rtol=1e-9
        @test x_chol ≈ x_ref rtol=1e-9
    end
end

@testset "Finalizer GC stress" begin
    # Repeatedly construct and discard factorizations; force GC to run finalizers
    # and ensure we don't double-free or crash.
    for _ in 1:50
        A = sparse(randn(10, 10) + 10 * I)
        M = randn(10, 10)
        Asym = sparse(M * M' + 10 * I)
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

@testset "Mixed element/index types interoperate" begin
    # Each (Tv, Ti) variant should work standalone in sequence (the @ccall
    # dispatch picks the correct cs_* symbol from libcxsparse based on the
    # generic-function method we defined).
    for Tv in (Float64, ComplexF64), Ti in (Int32, Int64)
        Adense = Tv <: Complex ?
            complex.(randn(6, 6), randn(6, 6)) + Tv(6) * I :
            randn(6, 6) + Tv(6) * I
        A = SparseMatrixCSC{Tv,Ti}(sparse(Adense))
        b = Tv <: Complex ?
            Vector{Tv}(complex.(randn(6), randn(6))) :
            Vector{Tv}(randn(6))
        F = cs_qr(A)
        x = F \ b
        @test norm(A * x - b) < 1e-8 * norm(b)
    end
end
