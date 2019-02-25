module TestEP
using GaussianEP, Test, LinearAlgebra

function simple_ep_test()
    N = 3
    factors = [FactorPrior(IntervalPrior(a,b)) for (a,b) in [(0,1),(0,1),(-2,2)]]
    idx = [[i] for i in 1:N]
    FG = FactorGraph(factors, idx, N)
    F=[1.0 -1.0]
    P=[I; F]
    av0 = Float64[1/2, 1/2, 0]
    va0 = Float64[1/12, 1/12, 1/6]
    state,status,iter,ε = expectation_propagation(FG, P, epsconv=1e-8)
    @test state.μ ≈ av0 atol=1e-5
    @test diag(state.Σ) ≈ va0 atol=1e-5
    @test status === :converged
end


simple_ep_test()

printstyled("All TestEP passed!\n",color=:green,bold=true)
end #end module
