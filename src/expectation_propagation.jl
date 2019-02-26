using Random, LinearAlgebra, ExtractMacro


function update!(old, new, ρ=0.0)
    r = norm(new - old, Inf);
    old .*= ρ;
    old .+= (1 - ρ) * new;
    return r
end

struct EPState{T <: Real}
     Σ   :: Matrix{T}
     μ   :: Vector{T}
     J   :: Vector{Matrix{T}}
     h   :: Vector{Vector{T}}
     Jc  :: Vector{Matrix{T}}
     hc  :: Vector{Vector{T}}
     FG  :: FactorGraph
end

eye(n) = Matrix(1.0I, n, n)

function EPState(FG::FactorGraph)
    d(a) = length(FG.idx[a])
    M,N = length(FG.idx), FG.N
    return EPState(eye(N), zeros(N),
                    [eye(d(a)) for a=1:M], [zeros(d(a)) for a=1:M],
                    [eye(d(a)) for a=1:M], [zeros(d(a)) for a=1:M],
                    FG)
end

function update!(state::EPState, ψ::Factor, a::Int, ρ::Real)
    @extract state : Σ μ J h FG
    ∂a = FG.idx[a]
    # J, h are cavity coeffs
    Jc = Σ[∂a, ∂a]\I .- J[a]
    hc = Σ[∂a, ∂a]\μ[∂a] .- h[a]
    # JJ, hh are moments
    hh, JJ = moments(ψ, hc, Jc)
    # JJ, hh are now total exponents
    JJ .= JJ\I
    hh .= JJ*hh
    # JJ - Jc, hh - hc are new approximated factors
    return max(update!(J[a], JJ .- Jc, ρ), update!(h[a], hh .- hc, ρ))
end

"""
    expectation_propagation(FG::FactorGraph, P::AbstractMatrix{T} = Diagonal(FG.N), d::Vector{T} = zeros(FG.N);
        maxiter::Int = 2000,
        callback = (x...)->nothing,
        damp::T = 0.9,
        epsconv::T = 1e-6,
        maxvar::T = 1e50,
        minvar::T = 1e-50,
        state::EPState{T} = EPState{T}(FG),
        inverter::Function = inv) where {T <: Real, P <: Prior}


EP for approximate inference of

``P(\\bf{x})=\\frac1Z \\prod_a ψ_{a}(x_a)``

Arguments:

* `A::Array{Term{T}}`: Gaussian Term (involving only x)

Optional Arguments:

* `P::AbstractMatrix{Prior}`: Projector
* `d::AbstractVector{T}`: Contant shift

Optional named arguments:

* `maxiter::Int = 2000`: maximum number of iterations
* `callback = (x...)->nothing`: your own function to report progress, see [`ProgressReporter`](@ref)
* `state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2])`: If supplied, all internal state is updated here
* `damp::T = 0.9`: damping parameter
* `epsconv::T = 1e-6`: convergence criterion
* `maxvar::T = 1e50`: maximum variance
* `minvar::T = 1e-50`: minimum variance
* `inverter::Function = inv`: inverter method

# Example

```jldoctest
julia> FG=FactorGraph([FactorPrior(IntervalPrior(a,b)) for (a,b) in [(0,1),(0,1),(-2,2)]], [[i] for i=1:3], 3)
3-element Array{IntervalPrior{Int64},1}:
 IntervalPrior{Int64}(0, 1)
 IntervalPrior{Int64}(0, 1)
 IntervalPrior{Int64}(-2, 2)

julia> P=[I; [1.0 -1.0]]

julia> res = expectation_propagation(FG, P)
```

Note on subspace restriction

P(x) ∝ ∫dz δ(x-Pz-d) ∏ₐψₐ(xₐ)
   x = Pz + d
Q(x) ∝ ∏ₐϕₐ(xₐ)
     ∝ exp(-½ xᵀAx + xᵀy)
     ∝ ∫dz δ(x-Pz-d) Q(z)
Q(z) ∝ exp(-½ (Pz+d)ᵀA(Pz+d) + (Pz-d)ᵀy)
     ∝ exp(-½ zᵀPᵀAPz - zᵀPᵀAd -½dᵀAdᵀ + (zᵀPᵀ-dᵀ)y)
     ∝ exp(-½ zᵀPᵀAPz + zᵀ(Pᵀ(y - Ad))
Σz = (PᵀAP)⁻¹
μz = (PᵀAP)⁻¹Pᵀ(y-Ad)
Σx = P(PᵀAP)⁻¹Pᵀ
μx = P*Σz + d
= P((PᵀAP)⁻¹Pᵀ(y-Ad))+d
= Σx(y-Ad)+d
"""
function expectation_propagation(FG::FactorGraph,
                                 P::AbstractArray{T} = Diagonal(ones(FG.N)),
                                 d::AbstractVector{T} = zeros(FG.N); # x = Pz+d
                                 maxiter::Int64 = 2000,
                                 callback = (x...)->nothing,
                                 ρ::Float64 = 0.9,
                                 epsconv::Float64 = 1e-6,
                                 inverter = inv,
                                 state::EPState = EPState(FG)) where {T<:Real}

    @extract state : Σ μ J h
    size(P,1) == FG.N || throw(ArgumentError("bad size of projector"))

    N, M = FG.N, length(FG.factors)
    A, y = zeros(N,N), zeros(N)
    ε = 0.0
    for iter = 1:maxiter
        A .= 0.0
        y .= 0.0
        for a in 1:M
            ∂a = FG.idx[a]
            A[∂a, ∂a] .+= J[a]
            y[∂a] .+= h[a]
        end
        Σ .= P*inverter(P'*A*P)*P'
        μ .= Σ*(y - A*d) .+ d
        ε = 0.0
        for a=1:M
            ε = max(ε, update!(state, FG.factors[a], a, ρ))
        end
        callback(state,iter,ε) != nothing && break
        ε < epsconv && return (state, :converged, iter, ε)
    end
    return (state, :unconverged, maxiter, ε)
end

