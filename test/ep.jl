module TestEP
using GaussianEP, Test, LinearAlgebra

function simple_ep_test()
    N = 3
    factors = [FactorInterval(a,b) for (a,b) in [(0,1),(0,1),(-2,2)]]
    idx = [[i] for i in 1:N]
    S = [1.0 -1.0 -1.0]
    FG = FactorGraph(factors, idx, S)
    av0 = Float64[1/2, 1/2, 0]
    va0 = Float64[1/12, 1/12, 1/6]
    state,status,iter,ε = expectation_propagation(FG, epsconv=1e-8)
    @test state.μ ≈ av0 atol=1e-5
    @test diag(state.Σ) ≈ va0 atol=1e-5
    @test status === :converged
end


simple_ep_test()

printstyled("All TestEP passed!\n",color=:green,bold=true)
end #end module
