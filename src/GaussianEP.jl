module GaussianEP

export expectation_propagation, Term, EPState, EPOut, EPState2
export Prior, IntervalPrior, SpikeSlabPrior, BinaryPrior, GaussianPrior, PosteriorPrior, QuadraturePrior, AutoPrior, ThetaPrior

using ExtractMacro, SpecialFunctions, LinearAlgebra

include("Term.jl")
include("priors.jl")
include("expectation_propagation.jl")
include("ProgressReporter.jl")
include("Factor.jl")
include("ep.jl")
include("FactorGauss.jl")

end # end module
