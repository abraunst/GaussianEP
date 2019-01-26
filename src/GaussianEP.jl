module GaussianEP

using ExtractMacro, SpecialFunctions

include("Term.jl")
include("priors.jl")
include("expectation_propagation.jl")
include("ProgressReporter.jl")

export expectation_propagation

end # end module
