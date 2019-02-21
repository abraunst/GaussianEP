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
    S   :: Vector{Matrix{T}}
    h   :: Vector{Vector{T}}
end

function EPState2(FG::FactorGraph)
    d(a) = length(FG.idx[a])
    M = length(FG.idx)
    return EPState(zeros(N,N), zeros(N),
                   [zeros(d(a),d[a]) for a=1:M],
                   [zeros(d[a]) for a=1:M])
end

function update!(state::EPState2, ψ::Factor, a::Int, ρ::Real, epsvar2::Real)
    @extract state : Sc, yc, S, y, Σ, μ
    ∂a = ψ.idx;
    S[a] .= (Σ[∂a, ∂a] + epsvar2 * I)\eye(length(∂a)) .- S[a]
    y[a] .= (Σ[∂a, ∂a] + epsvar2 * I)\μ[∂a] .- h[a]
    av, cov = moments(ψ[a], y[a], S[a])
    icov = inv(cov + epsvar * I)
    return max(update!(S[a], icov - S[a]), update!(y[a], icov * av - y[a]))
end

function EP(FG::FactorGraph,
            P::AbstractMatrix{T} = I,
            d::AbstractVector{T} = zeros(FG.N); # x = Fy+d
            maxiter::Int64 = 2000,
            callback = (x...)->nothing,
            ρ::Float64 = 0.9,
            epsconv::Float64 = 1e-6,
            epsvar::Float64 = 1e-10,
            epsvar2::Float64 = 0.0,
            inverter = inv,
            state::EPState2 = EPState2(FG))

    @extract state : Σ μ S h

    N, M = FG.N, length(FG.factors)
    A, y = zeros(N,N), zeros(N)
    Σ1 = zeros(size(P,2), size(P,2))
    for iter = 1:maxiter
        A[:] .= 0.0
        y .= 0.0
        for a in 1:M
            ψₐ, ∂a = FG.factors[a], FG.idx[a]
            A[∂a,∂a] .+= S[a]; y[∂a] .+= h[a]
        end
        Σ1 .= inverter(P'*A*P + epsvar * I)
      	Σ .= P'Σ1*P
        μ .= d .+ P*Σ1*(y - P'A*d)
        ε = maximum(update!(state, F.factors[a], a, ρ, epsvar2) for a=1:M)
        callback(state,iter,ε) != nothing && break
        ε < epsconv && return state, :converged, iter, ε
    end
    return state, :unconverged, iter, ε
end


function expectation_propagation2(H::Vector{Term{T}}, P0::Vector{P}, F::AbstractMatrix{T} = zeros(T,0,length(P0)), d::AbstractVector{T} = zeros(T,size(F,1));
                     x...) where {T <: Real, P <: Prior}
    F = FactorGraph(cat(Factor[FactorPrior(p) for p in P0], Factor[FactorGauss(h.A, h.y) for h in H]))
    EP(F, x...)
end
