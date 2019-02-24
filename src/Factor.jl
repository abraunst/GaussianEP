abstract type Factor
end

function moments(F::Factor, y::Vector{Float64}, S::Vector{Float64})
    throw(MethodError("Undefined moments"))
end


struct FactorGraph
    factors::Vector{Factor}
    idx::Vector{Vector{Int}}
    N::Int
end


struct FactorPrior{P<:Prior} <: Factor
    P0::P
    n::Int
end

FactorPrior(p::Prior) = FactorPrior(p,1)

moments(F::FactorPrior, y::Vector{Float64}, S::Matrix{Float64}) = ((m, v) = moments(F.P0, y[1]/S[1], 1/sqrt(S[1])); ([m],fill(v,1,1)))
