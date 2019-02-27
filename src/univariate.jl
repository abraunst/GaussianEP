using FastGaussQuadrature, ForwardDiff

export FactorInterval, FactorSpikeSlab, FactorBinary, FactorGaussian, FactorPosterior, FactorQuadrature, FactorAuto, FactorTheta


abstract type FactorUnivariate <: Factor end

ret((av,va)) = (fill(av,1), fill(va,1,1))

moments!(av::Vector, va::Matrix, ψ::FactorUnivariate, h, J) = (p = moments(ψ, h, J); av[]=p[1]; va[]=p[2]; return)


"""
Interval prior

Parameters: l,u

`` p_0(x) = \\frac{1}{u-l}\\mathbb{I}[l\\leq x\\leq u] ``
"""
struct FactorInterval{T<:Real} <: FactorUnivariate
    l::T
    u::T
end

Φ(x) = 0.5*(1+erf(x/sqrt(2.0)))
ϕ(x) = exp(-x.^2/2)/sqrt(2π)


function moments(p0::FactorInterval,h,J)
    J, h = J[], h[]
    if J <= 0
        return 0.5 * (xu + xl), (xu + xl)^2/12
    end
    σ = 1/sqrt(J)
    μ = σ*h
    xl = (p0.l - μ)/σ
    xu = (p0.u - μ)/σ
    minval = min(abs(xl), abs(xu))

    if xu - xl < 1e-10
        return 0.5 * (xu + xl), (xu + xl)^2/12
    end

    if minval <= 6.0 || xl * xu <= 0
        ϕu, Φu, ϕl, Φl  = ϕ(xu), Φ(xu), ϕ(xl), Φ(xl)
        av = (ϕl - ϕu) / (Φu - Φl)
        mom2 = (xl * ϕl - xu * ϕu) / (Φu - Φl)
    else
        Δ = (xu^2 - xl^2) * 0.5
        if Δ > 40.0
            av = xl^5 / (3.0 - xl^2 + xl^4)
            mom2 = xl^6 / (3.0 - xl^2 + xl^4)
        else
            eΔ = exp(Δ)
            av = (xl * xu)^5 * (1. - eΔ) / (-eΔ * (3.0 - xl^2 + xl^4) * xu^5 + xl^5 * (3.0 - xu^2 + xu^4))
            mom2 = (xl * xu)^5 * (xu - xl * eΔ) / (-eΔ * (3.0 - xl^2 + xl^4) * xu^5 + xl^5 * (3.0 - xu^2 + xu^4))
        end
    end
    va = mom2 - av^2
    return μ + av * σ, σ^2 * (1 + va)
end


"""
Spike-and-slab prior

Parameters: ρ,λ

`` p_0(x) = (1-ρ) δ(x) + ρ \\mathcal{N}(x;0,λ^{-1}) ``
"""
mutable struct FactorSpikeSlab{T<:Real} <: FactorUnivariate
    ρ::T
    λ::T
    δρ::T
    δλ::T
end


"""
``p = \\frac1{(ℓ+1)((1/ρ-1) e^{-\\frac12 (μ/σ)^2 (2-\\frac1{1+ℓ})}\\sqrt{1+\\frac1{ℓ}}+1)}``
"""
function moments(p0::FactorSpikeSlab,h,J)
    J, h = J[], h[]
#=
    s2 = σ^2
    d = 1 + p0.λ * s2;
    sd = 1 / (1/s2 + p0.λ);
    n = μ^2/(2*d*s2);
    Z = sqrt(p0.λ * sd) * p0.ρ;
    f = 1 + (1-p0.ρ) * exp(-n) / Z;
    av = μ / (d * f);
    va = (sd + (μ / d)^2 ) / f - av^2;
    #p0 = (1 - p0.params.ρ) * exp(-n) / (Z + (1-p0.params.ρ).*exp(-n));
    =#
    σ = 1/sqrt(J)
    μ = h/J
    ℓ0 = p0.λ * σ^2
    ℓ = 1 + ℓ0;
    z = ℓ * (1 + (1/p0.ρ-1) * exp(-0.5*(μ/σ)^2/ℓ) * sqrt(ℓ/ℓ0))
    return μ / z, (σ^2 + μ^2*(1/ℓ - 1/z)) / z;
end


function gradient(p0::FactorSpikeSlab, h, J)
    J, h = J[], h[]
    s = 1/J
    μ = h/J
    d = 1 + p0.λ * s;
    q = sqrt(p0.λ * s / d);
    f = exp(-μ^2 / (2s*d));
    den = (1 - p0.ρ) * f + q * p0.ρ;
    # update ρ
    if p0.δρ > 0
        p0.ρ += p0.δρ * (q - f) / den;
        p0.ρ = clamp(p0.ρ, 0, 1)
    end
    # update λ
    if p0.δλ > 0
        num = s * (1 + p0.λ * (s - μ^2)) / (2q * d^3) * p0.ρ;
        p0.λ += p0.δλ * num/den;
        p0.λ = max(p0.λ, 0)
    end
    nothing
end


"""
Binary Factor

p_0(x) ∝ ρ δ(x-x_0) + (1-ρ) δ(x-x_1)

"""
struct FactorBinary{T<:Real} <: FactorUnivariate
    x0::T
    x1::T
    ρ::T
end


function moments(p0::FactorBinary, h, J)
    J = J[1]; h = h[1]
    w = exp((-1/2*J*(p0.x0+p0.x1)+h)*(p0.x0-p0.x1))
    Z = p0.ρ *w + (1-p0.ρ);
    av = p0.ρ * p0.x0 * w + (1-p0.ρ) * p0.x1;
    mom2 = p0.ρ * (p0.x0^2) * w + (1-p0.ρ) * (p0.x1^2);
    if (isnan(Z) || isinf(Z))
        Z = p0.ρ + (1-p0.ρ) / w;
        av = p0.ρ * p0.x0 + (1-p0.ρ) * p0.x1 / w;
        mom2 = p0.ρ * (p0.x0^2) + (1-p0.ρ) * p0.x1 / w;
    end
    av /= Z;
    mom2 /= Z;
    va = mom2 - av.^2;
    return av, va
end


"""
This is a fake Factor that can be used to fix experimental moments
Parameters: μ, v (variance, not std)
"""
struct FactorPosterior{T<:Real} <: FactorUnivariate
    μ::T
    v::T
end

function moments(p0::FactorPosterior, h, J)
    return p0.μ,p0.v
end


struct FactorQuadrature{T<:Real} <: FactorUnivariate
    f
    X::Vector{T}
    W0::Vector{T}
    W1::Vector{T}
    W2::Vector{T}
    function (FactorQuadrature)(f; a::T=-1.0, b::T=1.0, points::Int64=1000) where {T<:Real}
        X,W = gausslegendre(points)
        X = 0.5*(X*(b-a).+(b+a))
        W .*= 0.5*(b-a)
        W0 = map(f,X) .* W;
        W1 = X .* W0;
        W2 = X .* W1;
        new{T}(f,X,W0,W1,W2)
    end
end

function moments(p0::FactorQuadrature, h, J)
    J = J[]; h = h[]
    v = map(x->exp(-J/2*x^2+h*x),p0.X)
    z0 = v ⋅ p0.W0
    av = (v ⋅ p0.W1)/z0
    va = (v ⋅ p0.W2)/z0 - av^2
    return av,va
end

mutable struct FactorAuto{T<:Real} <: FactorUnivariate
    #real arguments
    f
    P::Vector{T}
    dP::Vector{T}
    #quadrature data
    X::Vector{T}
    W::Vector{T}
    #cached data (updated in ctor and update!)
    X2::Vector{T}
    oldP::Vector{T}
    FXW::Vector{T}
    DFXW::Matrix{T}
    cfg::Any
    function (FactorAuto)(f, P::Vector{T}, dP::Vector{T} = zeros(length(P)), a=-1, b=1, points=1000) where {T<:Real}
        X,W = gausslegendre(points)
        X = 0.5*(X*(b-a).+(b+a))
        W .*= 0.5*(b-a)
        np = length(P)
        new{T}(f,P,dP,X,W,X.^2,fill(Inf,np),fill(zero(T),points),fill(zero(T),np,points),ForwardDiff.GradientConfig(f,[X[1];P]))
    end
end

function moments(p0::FactorAuto, h, J)
    do_update!(p0)
    J = J[1]; h = h[1]
    v = p0.FXW .* map(x->exp(-J/2 * x^2 + h*x), p0.X)
    v .*= 1/sum(v)
    av = v ⋅ p0.X
    va = (v ⋅ p0.X2) - av^2
    return av,va
end

function do_update!(p0::FactorAuto)
    p0.P == p0.oldP && return
    copy!(p0.FXW, p0.f.([[x;p0.P] for x in p0.X]) .* p0.W)
    for i in 1:length(p0.X)
        p0.DFXW[:,i] = ForwardDiff.gradient(p0.f,[p0.X[i];p0.P],p0.cfg)[2:end]
    end
    copy!(p0.oldP, p0.P)
end

function gradient(p0::FactorAuto, h, J)
    J = J[]; h = h[]
    s22 = 2/J
    μ = h/J
    v = map(x->exp(-(x-μ)^2 / s22), p0.X)
    z = sum(v)
    v ./= z
    p0.P .+= p0.dP .* (p0.DFXW * v)
end


###### Thomas's theta moments

#The quotient of normal PDF/CF with the proper expansion
#The last expansion term should not be necessary though.
function pdf_cf(x)
    if x < -8
        return -1/x-x+2x^(-3)-10x^(-5)
    end
    return sqrt(2/pi)*exp(-x^2/2)/(1+erf(x/sqrt(2)))
end

"""
A θ(x) prior
"""
struct FactorTheta <: FactorUnivariate end


function moments(::FactorTheta,h,J)
    J, h= J[], h[]
    μ = h/J
    α = h/sqrt(J)
    av = μ+pdf_cf(α)/sqrt(J)
    va = 1/J*(1-α*pdf_cf(α)-pdf_cf(α)^2)
    return av,va
end

