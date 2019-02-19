module TestEP
using GaussianEP, Test

function simple_ep_test()
    t=Term(zeros(2,2),zeros(2),1.0)
    P=[IntervalPrior(i...) for i in [(0,1),(0,1),(-2,2)]]
    F=[1.0 -1.0]
    av0 = [0.4999974709003177,0.4999974709003177,3.665273196564082e-15]
    va0 = [0.08332501737195087, 0.08332501737195087, 0.2043006364495929]
    μ0 = [0.4898618134668008,0.4898618134668008,3.665993043660408e-15]
    s0 = [334.0179053087342,334.0179053087342,0.20434113796777062]
    res = expectation_propagation([t], P, F)
    @test sum(abs, res.av - av0) < 1e-12
    @test sum(abs, res.va - va0) < 1e-12
    @test sum(abs, res.μ - μ0) < 1e-12
    @test sum(abs, res.s - s0) < 1e-12    
end

simple_ep_test()

printstyled("All TestEP passed!\n",color=:green,bold=true)
end #end module
