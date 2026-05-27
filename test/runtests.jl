using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "Core" || GROUP == "QR"
        @safetestset "cs_qr" include("qr_tests.jl")
    end

    if GROUP == "All" || GROUP == "Core" || GROUP == "LU"
        @safetestset "cs_lu" include("lu_tests.jl")
    end

    if GROUP == "All" || GROUP == "Core" || GROUP == "Cholesky"
        @safetestset "cs_cholesky" include("cholesky_tests.jl")
    end

    if GROUP == "All" || GROUP == "Core" || GROUP == "Cross"
        @safetestset "cross-factorization" include("cross_tests.jl")
    end

    if GROUP == "All" || GROUP == "QA"
        @safetestset "Quality Assurance" include("qa.jl")
    end
end
