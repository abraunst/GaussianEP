module GaussianEP

using ExtractMacro, SpecialFunctions, LinearAlgebra


include("Term.jl")
include("priors.jl")
include("expectation_propagation.jl")
include("ProgressReporter.jl")

export expectation_propagation, Term
export Prior, IntervalPrior, SpikeSlabPrior, BinaryPrior, GaussianPrior, PosteriorPrior, QuadraturePrior, AutoPrior, ThetaPrior

end # end module
