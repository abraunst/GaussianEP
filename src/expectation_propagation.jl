using Random, LinearAlgebra, ExtractMacro


function update!(old::Array{T}, new::Array{T}, ρ::T = zero(T))::T where {T<:Real}
    r = maximum(abs, new - old)
    old .*= ρ
    old .+= (1 - ρ) * new
    return r
end

struct EPState{T <: Real, F <: Factor}
     Σ   :: Matrix{T}
     μ   :: Vector{T}
     J   :: Vector{Matrix{T}}
     h   :: Vector{Vector{T}}
     Jc  :: Vector{Matrix{T}}
     hc  :: Vector{Vector{T}}
     Jt  :: Vector{Matrix{T}}
     ht  :: Vector{Vector{T}}
     FG  :: FactorGraph{T,F}
end

eye(::Type{T}, n::Integer) where T = Matrix(T(1)*I, n, n)

function EPState(FG::FactorGraph{T,F}) where {T <: Real, F <: Factor}
    d(a) = length(FG.idx[a])
    M,N = length(FG.idx), FG.N
    return EPState{T,F}(eye(T, N), zeros(T, N),
                    [eye(T, d(a)) for a=1:M], [zeros(T, d(a)) for a=1:M],
                    [eye(T, d(a)) for a=1:M], [zeros(T, d(a)) for a=1:M],
                    [eye(T, d(a)) for a=1:M], [zeros(T, d(a)) for a=1:M],
                    FG)
end

function update!(state::EPState{T}, ψ::Factor, a::Integer, ρ::T, epsvar::T = zero(T)) where {T <: Real}
    @extract state : Σ μ J h Jc hc Jt ht FG
    ∂a = FG.idx[a]
    # J, h are cavity coeffs
    hca, Jca = hc[a], Jc[a]
    Jca .= (Σ[∂a, ∂a]+epsvar*I)\I .- J[a]
    hca .= Σ[∂a, ∂a]\μ[∂a] .- h[a]
    # Jta, hta are moments
    hta, Jta = ht[a], Jt[a]
    moments!(hta, Jta, ψ, hca, Jca)
    # Jta, hta are now total exponents
    Jta .= (Jta+epsvar*I)\I
    hta .= Jta*hta
    # Jta - Jc, hta - hc are new approximated factors
    ε = max(update!(J[a], Jta .- Jca, ρ), update!(h[a], hta .- hca, ρ))
    # learn params
    learn!(ψ, hca, Jca)
    return ε
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
* `damp::T = 0.0`: damping parameter
* `epsconv::T = 1e-6`: convergence criterion
* `maxvar::T = 1e50`: maximum variance
* `minvar::T = 1e-50`: minimum variance
* `inverter::Function = inv`: inverter method

# Example

```jldoctest
julia> FG=FactorGraph([FactorInterval(a,b) for (a,b) in [(0,1),(0,1),(-2,2)]], [[i] for i=1:3], [1.0 -1.0 -1.0])
FactorGraph(Factor[FactorInterval{Int64}(0, 1), FactorInterval{Int64}(0, 1), FactorInterval{Int64}(-2, 2)], Array{Int64,1}[[1], [2], [3]], 3)

julia> using LinearAlgebra

julia> res = expectation_propagation(FG)
(EPState{Float64}([0.0833329 1.00114e-6 0.0833319; 1.00114e-6 0.0833329 -0.0833319; 0.0833319 -0.0833319 0.166664], [0.499994, 0.499994, 1.39058e-13], Array{Float64,2}[[11.9999], [11.9999], [0.00014416]], Array{Float64,1}[[5.99988], [5.99988], [-1.14443e-13]], Array{Float64,2}[[1.0], [1.0], [1.0]], Array{Float64,1}[[0.0], [0.0], [0.0]], FactorGraph(Factor[FactorInterval{Int64}(0, 1), FactorInterval{Int64}(0, 1), FactorInterval{Int64}(-2, 2)], Array{Int64,1}[[1], [2], [3]], 3)), :converged, 162, 9.829257408000558e-7)
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
function expectation_propagation(FG::FactorGraph{T,F};
                                 maxiter::Integer = 2000,
                                 callback = (x...)->nothing,
                                 damp::T = zero(T),
                                 epsconv::T = 1e-6,
                                 inverter = inv,
                                 epsvar::T = zero(T),
                                 state::EPState{T} = EPState(FG)) where {F<:Factor, T<:Real}

    @extract state : Σ μ J h
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
        Σ .= FG.P*inverter(FG.P'*A*FG.P)*FG.P'
        μ .= Σ * (y .- A*FG.d) .+ FG.d
        ε = 0.0
        for a=1:M
            ε = max(ε, update!(state, FG.factors[a], a, damp, epsvar))
        end
        callback(state,iter,ε) != nothing && break
        ε < epsconv && return (state, :converged, iter, ε)
    end
    return (state, :unconverged, maxiter, ε)
end

