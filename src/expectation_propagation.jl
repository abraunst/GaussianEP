using Random

function update_err!(dst, i, val)
    r=abs(val - dst[i])
    dst[i] = val
    return r
end

struct EPState{T}
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

EPState{T}(N, Nx) where {T <: Real} = EPState{T}(Matrix{T}(undef,Nx,Nx), zeros(T,Nx), Matrix{T}(undef,Nx,Nx), zeros(T,Nx),zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N), ones(T,N), ones(T,N))


"""
EP for

``P(\\bf{x})=\\frac1Z exp(-\\frac12\\bf{x}' A \\bf{x} + \\bf{x'} \\bf{y}))×\\prod_i p_{i}(x_i)``

Mandatory arguments:

* A::Array{Term},
* P0::Array{Prior}

* If F::Array{Real,2} with is included, the unknown becomes ``(\\bf{x},\\bf{y})^T`` and a term ``\\delta(F \\bf{x}+\bf{d}-\\bf{y})`` is added.
"""
function EP(H::Vector{Term{T}}, P0::Vector{P}, F::AbstractMatrix{T} = Matrix{T}(undef,0,size(P0,1)), d::AbstractVector{T} = zeros(T,size(F,1));
                     maxiter::Int = 2000,
                     callback = (x...)->nothing,
                     state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),
                     damp::T = 0.9,
                     epsconv::T = 1e-6,
                     maxvar::T = 1e50,
                     minvar::T = 1e-50,
                     inv::Function = inv) where {T <: Real, P <: Prior}
    @extract state A y Σ v av va a μ b s
    Ny,Nx = size(F)
    N = Nx + Ny
    @assert size(P0,1) == N
    Fp = copy(F')
    for iter = 1:maxiter
        sum!(A,y,H)
        Δμ, Δs, Δav, Δva = 0.0, 0.0, 0.0, 0.0
        A .+= Diagonal(1 ./ b[1:Nx]) .+ Fp * Diagonal(1 ./ b[Nx+1:end]) * F
        Σ .= inv(A)
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
                ss == b[i] && warn("infinite var, ss = ", ss)
                Δs = max(Δs, update_err!(s, i, maxvar))
                Δμ = max(Δμ, update_err!(μ, i, 0))
            end
            tav, tva = moments(P0[i], μ[i], sqrt(s[i]));
            Δav = max(Δav, update_err!(av, i, tav))
            Δva = max(Δva, update_err!(va, i, tva))
            (isnan(av[i]) || isnan(va[i])) && warn("avnew = $(av[i]) varnew = $(va[i])")

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
            return av, va, μ, s, :converged
        end
    end
    return av, va, μ, s, :unconverged
end
