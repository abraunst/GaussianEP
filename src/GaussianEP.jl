module GaussianEP

export FactorGraph, FactorPrior, FactorGauss, EPState, expectation_propagation

using ExtractMacro, SpecialFunctions, LinearAlgebra

include("ProgressReporter.jl")
include("Factor.jl")
include("FactorGraph.jl")
include("expectation_propagation.jl")
include("univariate.jl")
include("multivariate.jl")


end # end module
