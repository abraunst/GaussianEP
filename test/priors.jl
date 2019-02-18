module TestPrior
using GaussianEP,Test

# test spike and slab

spikeandslabmom(μ,σ,ρ,λ) = GaussianEP.moments(SpikeSlabPrior(ρ,λ,0.0,0.0),μ,σ)

function spike_and_slab_test()
    @test isapprox.(spikeandslabmom(-1.9,13.0,1.0,0.12), (-0.0892857142857146,7.94172932330828),atol=1e-12) == (true,true)
    @test isapprox.(spikeandslabmom(0.0,13.0,0.0,0.12), (0.0,0.0)) == (true,true)
    @test isapprox.(spikeandslabmom(6.0,2.0,0.2,2.0), (0.1865693402764403,0.2139510016374009),atol=1e-12) == (true,true)
end


spike_and_slab_test()
printstyled("All TestPrior tests passed!\n",color=:green,bold=true)
end #end module
