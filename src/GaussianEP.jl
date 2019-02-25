module GaussianEP

export Factor, FactorGraph, FactorPrior, FactorGauss, EPState, expectation_propagation
export Prior, IntervalPrior, SpikeSlabPrior, BinaryPrior, GaussianPrior, PosteriorPrior, QuadraturePrior, AutoPrior, ThetaPrior

using ExtractMacro, SpecialFunctions, LinearAlgebra

include("ProgressReporter.jl")
include("priors.jl")
include("Factor.jl")
include("expectation_propagation.jl")
include("FactorGauss.jl")

end # end module
