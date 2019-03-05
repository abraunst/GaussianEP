
"""
This type represents an interaction term in the energy function of the form

``β_i (\\frac12 x'Ax + x'y + c) + M_i \\log β_i``

The complete energy function is given by

``∑_i β_i (\\frac12 x' A_i x + x' y_i + c_i) + M_i log β_i``

as is represented by an Vector{Term}. Note that c and M are only needed for paramenter learning
"""
mutable struct Term{T <: Real}
    A::Matrix{T}
    y::Vector{T}
    c::T
    β::T
    # for parameter learning
    δβ::T
    M::Int
end

Term(A,y,β = 1.0) = Term(A,y,0.0,β,0.0,0)

function (t::Term)(v::Vector)
    return v⋅(t.A*v-2*t.y) + t.c
end

function updateβ(t::Term{T}, v) where T
    if t.δβ > 0
        t.β = t.δβ * t.M / t(v) + (1-t.δβ) * t.β
    end
end

function sum!(A::Matrix{T}, y::Vector{T}, H::Vector{Term{T}}) where T <: Real
    fill!(A, zero(T))
    fill!(y, zero(T))
    for i=1:length(H)
        A .+= H[i].β * H[i].A
        y .+= H[i].β * H[i].y
    end
end
function update_err!(dst, i, val)
    r=abs(val - dst[i])
    dst[i] = val
    return r
end

"""
    Instantaneous state of an expectation propagation run.
"""
struct EPState{T<:AbstractFloat}
    A::Matrix{T}
    y::Vector{T}
    Σ::Matrix{T}
    v::Vector{T}
    av::Vector{T}
    va::Vector{T}
    a::Vector{T}
    μ::Vector{T}
    b::Vector{T}
    s::Vector{T}
end
EPState{T}(N, Nx = N) where {T <: AbstractFloat} = EPState{T}(Matrix{T}(undef,Nx,Nx), zeros(T,Nx), Matrix{T}(undef,Nx,Nx), zeros(T,Nx),zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N), ones(T,N), ones(T,N))

"""
Output of EP algorithm

"""
struct EPOut{T<:AbstractFloat}
    av::Vector{T}
    va::Vector{T}
    μ::Vector{T}
    s::Vector{T}
    converged::Symbol
    state::EPState{T}
end
function EPOut(s, converged::Symbol) where {T <: AbstractFloat}
    converged ∈ (:converged,:unconverged) || error("$converged is not a valid symbol")
    return EPOut(s.av,s.va, s.μ,s.s,converged,s)
end
function expectation_propagation_legacy(H::Vector{Term{T}}, P0::Vector{P}, F::AbstractMatrix{T} = zeros(T,0,length(P0)), d::AbstractVector{T} = zeros(T,size(F,1));
                     maxiter::Int = 2000,
                     callback = (x...)->nothing,
                     state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),
                     damp::T = 0.9,
                     epsconv::T = 1e-6,
                     maxvar::T = 1e50,
                     minvar::T = 1e-50,
                     inverter::Function = inv) where {T <: Real, P <: Prior}
    @extract state A y Σ v av va a μ b s
    Ny,Nx = size(F)
    N = Nx + Ny
    @assert size(P0,1) == N
    Fp = copy(F')
    for iter = 1:maxiter
        sum!(A,y,H)
        Δμ, Δs, Δav, Δva = 0.0, 0.0, 0.0, 0.0
        A .+= Diagonal(1 ./ b[1:Nx]) .+ Fp * Diagonal(1 ./ b[Nx+1:end]) * F
        Σ .= inverter(A)
        v .= Σ * (y .+ a[1:Nx] ./ b[1:Nx] .+ Fp * ((a[Nx+1:end]-d) ./ b[Nx+1:end]))
        for i in 1:N
            if i <= Nx
                ss = clamp(Σ[i,i], minvar, maxvar)
                vv = v[i]
            else
                x = Fp[:, i-Nx]
                ss = clamp(dot(x, Σ*x), minvar, maxvar)
                vv = dot(x, v) + d[i-Nx]
            end

            if ss < b[i]
                Δs = max(Δs, update_err!(s, i, clamp(1/(1/ss - 1/b[i]), minvar, maxvar)))
                Δμ = max(Δμ, update_err!(μ, i, s[i] * (vv/ss - a[i]/b[i])))
            else
                ss == b[i] && @warn "infinite var, ss = $ss"
                Δs = max(Δs, update_err!(s, i, maxvar))
                Δμ = max(Δμ, update_err!(μ, i, 0))
            end
            tav, tva = moments(P0[i], μ[i], sqrt(s[i]));
            Δav = max(Δav, update_err!(av, i, tav))
            Δva = max(Δva, update_err!(va, i, tva))
            (isnan(av[i]) || isnan(va[i])) && @warn "avnew = $(av[i]) varnew = $(va[i])"

            new_b = clamp(1/(1/va[i] - 1/s[i]), minvar, maxvar)
            new_a = av[i] + new_b * (av[i] - μ[i])/s[i]
            a[i] = damp * a[i] + (1 - damp) * new_a
            b[i] = damp * b[i] + (1 - damp) * new_b
        end

        # learn prior's params
        for i in randperm(N)
            gradient(P0[i], μ[i], sqrt(s[i]));
        end
        # learn β params
        for i in 1:length(H)
            updateβ(H[i], av[1:Nx])
        end
        callback(av,Δav,epsconv,maxiter,H,P0)
        if Δav < epsconv
            return EPOut(state, :converged)
        end
    end
    return EPOut(state, :unconverged)
end
