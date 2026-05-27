using CXSparse
using Aqua
using ExplicitImports
using Test

@testset "Aqua" begin
    Aqua.test_all(CXSparse; ambiguities = false, piracies = false)
    # Ambiguities are checked package-locally to avoid noise from deps like
    # SparseArrays exposing ambiguous fallbacks not under our control.
    Aqua.test_ambiguities(CXSparse; recursive = false)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(CXSparse) === nothing
    @test check_no_stale_explicit_imports(CXSparse) === nothing
    @test check_all_qualified_accesses_via_owners(CXSparse) === nothing
end
