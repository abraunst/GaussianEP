using ExtractMacro

eye(N) = Matrix(1.0I, N, N)

function update!(old, new, ρ=0.0)
    r = norm(new - old, Inf);
    old .*= ρ;
    old .+= (1 - ρ) * new;
    return r
end

"""
EP
"""
struct EPState2{T <: Real}
    Σ   :: Matrix{T}
    μ   :: Vector{T}
    J   :: Vector{Matrix{T}}
    h   :: Vector{Vector{T}}
    FG  :: FactorGraph
end

function EPState2(FG::FactorGraph)
    d(a) = length(FG.idx[a])
    M,N = length(FG.idx), FG.N
    return EPState2(eye(N), zeros(N),
                    [eye(d(a)) for a=1:M], [zeros(d(a)) for a=1:M],
                    FG)
end

function update!(state::EPState2, ψ::Factor, a::Int, ρ::Real)
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
function EP(FG::FactorGraph,
            P::AbstractArray{T} = eye(FG.N),
            d::AbstractVector{T} = zeros(FG.N); # x = Pz+d
            maxiter::Int64 = 2000,
            callback = (x...)->nothing,
            ρ::Float64 = 0.9,
            epsconv::Float64 = 1e-6,
            inverter = inv,
            state::EPState2 = EPState2(FG)) where {T<:Real}

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


function expectation_propagation2(H::Vector{Term{T}}, P0::Vector{P}, F::AbstractArray{Float64} = zeros(0,length(P0))) where {T <: Real, P <: Prior}
    N = length(P0)
    factors = vcat(Factor[FactorPrior(p) for p in P0], Factor[FactorGauss(t.A, t.y) for t in H])
    idx = vcat([[i] for i in 1:N], [collect(1:N) for h in H])
    EP(FactorGraph(factors, idx, N), [I;F], callback=(state,iter,ε)->println("$iter $ε $(state.h)"))
end
