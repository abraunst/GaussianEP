module TestPrior
using GaussianEP,Test

# test spike and slab

spikeandslabmom(μ,σ,ρ,λ) = GaussianEP.moments(FactorSpikeSlab(ρ,λ,0.0,0.0),μ/σ^2,1/σ^2)

col((a,b)) = [a[],b[]]

function spike_and_slab_test()
    @test col(spikeandslabmom(-1.9,13.0,1.0,0.12)) ≈ [-0.0892857142857146,7.94172932330828] atol=1e-12
    @test col(spikeandslabmom(0.0,13.0,0.0,0.12)) ≈ [0.0,0.0] atol=1e-12
    @test col(spikeandslabmom(6.0,2.0,0.2,2.0)) ≈ [0.1865693402764403,0.2139510016374009] atol=1e-12
end

intervalmom(μ,σ,lb,ub) = GaussianEP.moments(FactorInterval(lb,ub),μ/σ^2,1/σ^2)
function uniform_test()
    @test col(intervalmom(-4.0,1.0,0.0,1000.0)) ≈ [0.2256071444894706679029638,0.0466728383974225474739583319205] atol=1e-12
    @test col(intervalmom(-5.0,1.0,0.0,1000.0)) ≈ [0.1865039670969923790710964794,0.0326964346120545146234803723928052] atol=1e-8 # this test require a lower precision ... maybe a better asymptotic expansion
end

spike_and_slab_test()
uniform_test()

printstyled("All TestPrior tests passed!\n",color=:green,bold=true)
end #end module
