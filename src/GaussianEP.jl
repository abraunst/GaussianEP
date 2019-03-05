module GaussianEP

using ExtractMacro, SpecialFunctions, LinearAlgebra

export FactorGraph, FactorGauss, EPState, expectation_propagation

include("ProgressReporter.jl")
include("Factor.jl")
include("FactorGraph.jl")
include("expectation_propagation.jl")
include("univariate.jl")
include("multivariate.jl")


end # end module
