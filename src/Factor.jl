abstract type Factor
end

function moments(F::Factor, y::Vector, S::Vector)
    throw(MethodError("Undefined moments"))


struct FactorGraph
    factors::Vector{Factor}
    idx::Vector{Vector{Int}}
    N::Int
end


struct FactorFunction <: Factor
    E::Function
end


"""
This function should compute first and second moments of exp(-F.E(x) -½ x'*S*x + x'*y)
"""
function moments(F::Factor, y::Vector{Float64}, S::Matrix{Float64})
    n = length(F.idx)
    N = 1<<n
    av = zeros(n)
    cov = zeros(n,n)
    z = 0
    X = [ [2*((i >> k)%2)-1 for k in 0:(n-1)] for i=0:(N-1) ]
    Y = [ -F.E(X[i]) + X[i]'*(y - 0.5*S*X[i]) for i=1:N ]
    m = maximum(Y)
    @inbounds for i=1:N
        px = exp(Y[i]-m)
        @assert(px>=0)
        z += px
        av .+= X[i] * px
        cov .+= (X[i]*X[i]') * px
    end
    av ./= z
    cov ./= z
    cov .-= av*av'
    return av, cov
end

struct FactorGauss <: Factor
   J::Matrix{Float64}
   h::Matrix{Float64}
   E::Fuction
end

FactorGauss(J,h) = FactorGauss(J, h, x->x'(h-0.5*J*x))

function update!(state::EPState, ψ::FactorGauss, a::Int, ρ::Real, epsvar2::Real)
    @extract state : Sc, yc, S, y, Σ, μ
    S[a] .= J
    y[a] .= h
    return 0.0
end


struct FactorPrior{P<:Prior}
    P0::Prior
end

moments(F::FactorPrior, y, S) = (Σ = inv(S); moments(F.P0, Σ*y, Σ))
