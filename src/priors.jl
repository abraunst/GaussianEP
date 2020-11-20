using FastGaussQuadrature, ForwardDiff

Φ(x) = 0.5*(1+erf(x/sqrt(2.0)))
ϕ(x) = exp(-x.^2/2)/sqrt(2π)

"""
Abstract Univariate Prior type
"""
abstract type Prior end

"""
    moments(p0::T, μ, σ) where T <:Prior -> (mean, variance)

    input: ``p_0, μ, σ``

    output: mean and variance of

    `` p(x) ∝ p_0(x) \\mathcal{N}(x;μ,σ) ``
"""
function moments(p0::T, μ, σ) where T <: Prior
    error("undefined moment calculation, assuming uniform prior")
    return μ,σ^2
end

"""

    gradient(p0::T, μ, σ) -> nothing

    update parameters with a single learning gradient step (learning rate is stored in p0)
"""
function gradient(p0::T, μ, σ) where T <: Prior
    #by default, do nothing
    return
end

"""
Interval prior

Parameters: l,u

`` p_0(x) = \\frac{1}{u-l}\\mathbb{I}[l\\leq x\\leq u] ``
"""
struct IntervalPrior{T<:Real} <: Prior
    l::T
    u::T
end

function moments(p0::IntervalPrior,μ,σ)
    xl = (p0.l - μ)/σ
    xu = (p0.u - μ)/σ
    minval = min(abs(xl), abs(xu))

    if xu - xl < 1e-10
        return 0.5 * (xu + xl), -1
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

`` p_0(x) ∝ (1-ρ) δ(x) + ρ \\mathcal{N}(x;0,λ^{-1}) ``
"""
mutable struct SpikeSlabPrior{T<:Real} <: Prior
    ρ::T
    λ::T
    δρ::T
    δλ::T
end

SpikeSlabPrior(ρ,λ) = SpikeSlabPrior(ρ,λ,0.0,0.0);

"""
``p = \\frac1{(ℓ+1)((1/ρ-1) e^{-\\frac12 (μ/σ)^2 (2-\\frac1{1+ℓ})}\\sqrt{1+\\frac1{ℓ}}+1)}``
"""
function moments(p0::SpikeSlabPrior,μ,σ)
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
    ℓ0 = p0.λ * σ^2
    ℓ = 1 + ℓ0;
    z = ℓ * (1 + (1/p0.ρ-1) * exp(-0.5*(μ/σ)^2/ℓ) * sqrt(ℓ/ℓ0))
    av = μ / z;
    va = (σ^2 + μ^2*(1/ℓ - 1/z)) / z;
    return av, va
end


function gradient(p0::SpikeSlabPrior, μ, σ)
    s = σ^2
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
end


"""
Binary Prior

p_0(x) ∝ ρ δ(x-x_0) + (1-ρ) δ(x-x_1)

"""
struct BinaryPrior{T<:Real} <: Prior
    x0::T
    x1::T
    ρ::T
end


function moments(p0::BinaryPrior, μ, σ)
    arg = -(σ^2 / 2) * (-p0.x0^2 - 2*(p0.x1 -p0.x0) * μ + p0.x1^2);
    earg = exp(arg)
    Z = p0.ρ / earg + (1-p0.ρ);
    av = p0.ρ * p0.x0 / earg + (1-p0.ρ) * p0.x1;
    mom2 = p0.ρ * (p0.x0^2) / earg + (1-p0.ρ) * (p0.x1^2);
    if (isnan(Z) || isinf(Z))
        Z = p0.ρ + (1-p0.ρ) * earg;
        av = p0.ρ * p0.x0 + (1-p0.ρ) * p0.x1 * earg;
        mom2 = p0.ρ * (p0.x0^2) + (1-p0.ρ) * p0.x1 * earg;
    end
    av /= Z;
    mom2 /= Z;
    va = mom2 - av.^2;
    return av,va
end


struct GaussianPrior{T<:Real} <: Prior
    μ::T
    β::T
    δβ::T
end

function moments(p0::GaussianPrior, μ, σ)
    s = 1/(1/σ^2 + p0.β)
    return s*(μ/σ^2 + p0.μ * p0.β), s
end

"""
This is a fake Prior that can be used to fix experimental moments
Parameters: μ, v (variance, not std)
"""
struct Posterior2Prior{T<:Real} <: Prior
    μ::T
    v::T
end

function moments(p0::Posterior2Prior, μ, σ)
    return p0.μ,p0.v
end

struct Posterior1Prior{T<:Real} <: Prior
    μ::T
end

function moments(p0::Posterior1Prior, μ, σ)
    return p0.μ,σ^2
end


struct QuadraturePrior{T<:Real} <: Prior
    f
    X::Vector{T}
    W0::Vector{T}
    W1::Vector{T}
    W2::Vector{T}
    function (QuadraturePrior)(f; a::T=-1.0, b::T=1.0, points::Int64=1000) where {T<:Real}
        X,W = gausslegendre(points)
        X = 0.5*(X*(b-a).+(b+a))
        W .*= 0.5*(b-a)
        W0 = map(f,X) .* W;
        W1 = X .* W0;
        W2 = X .* W1;
        new{T}(f,X,W0,W1,W2)
    end
end

function moments(p0::QuadraturePrior, μ, σ)
    v = map(x->exp(-(x-μ)^2/2σ^2),p0.X)
    z0 = v ⋅ p0.W0
    av = (v ⋅ p0.W1)/z0
    va = (v ⋅ p0.W2)/z0 - av^2
    return av, va
end

mutable struct AutoPrior{T<:Real} <: Prior
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
    function (AutoPrior)(f, P::Vector{T}, dP::Vector{T} = zeros(length(P)), a=-1, b=1, points=1000) where {T<:Real}
        X,W = gausslegendre(points)
        X = 0.5*(X*(b-a).+(b+a))
        W .*= 0.5*(b-a)
        np = length(P)
        new{T}(f,P,dP,X,W,X.^2,fill(Inf,np),fill(zero(T),points),fill(zero(T),np,points),ForwardDiff.GradientConfig(f,[X[1];P]))
    end
end

function moments(p0::AutoPrior, μ, σ)
    do_update!(p0)
    s22 = 2σ^2
    v = p0.FXW .* map(x->exp(-(x-μ)^2 / s22), p0.X)
    v .*= 1/sum(v)
    av = v ⋅ p0.X
    va = (v ⋅ p0.X2) - av^2
    return av, va
end

function do_update!(p0::AutoPrior)
    p0.P == p0.oldP && return
    copy!(p0.FXW, p0.f.([[x;p0.P] for x in p0.X]) .* p0.W)
    for i in 1:length(p0.X)
        p0.DFXW[:,i] = ForwardDiff.gradient(p0.f,[p0.X[i];p0.P],p0.cfg)[2:end]
    end
    copy!(p0.oldP, p0.P)
end

function gradient(p0::AutoPrior, μ, σ)
    s22 = 2σ^2
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
struct ThetaPrior <: Prior end


function moments(::ThetaPrior,μ,σ)
    α=μ/σ
    av=μ+pdf_cf(α)*σ
    var=σ^2*(1-α*pdf_cf(α)-pdf_cf(α)^2)
    return av,var
end

"""
A mixture of theta priors: p_0(x)=η*Θ(x)+(1-η)*Θ(-x)
"""
mutable struct ThetaMixturePrior{T<:Real} <: Prior
    η::T
    δη::T
end

function theta_mixt_factor(x,η)
     if abs(x)<=Inf
         f=exp(-0.5*x^2.0)/(η*erfc(-sqrt(0.5)*x)+(1.0-η)*erfc(sqrt(0.5)*x))
     else
         println("In theta_mixt_factor sono *qui*!")
         if η!=0.0
             f=0.5*exp(-0.5*x^2.0)/η
         else
             f=sqrt(0.5*π)/(1.0/x-1.0/x^3.0+3.0/x^5.0)
         end
     end
end

##############
mutable struct ThetaMixturePrior2{T<:Real} <: Prior
    η::T
    δη::T
    thr::T
end

function theta_mixt_factor2(x,η,thr)
     if abs(x)<=thr
         f=exp(-0.5*x^2.0)/(η*erfc(-sqrt(0.5)*x)+(1.0-η)*erfc(sqrt(0.5)*x))
     else
         println("In theta_mixt_factor sono *qui*!")
         if η!=0.0
             f=0.5*exp(-0.5*x^2.0)/η
         else
             f=sqrt(0.5*π)/(1.0/x-1.0/x^3.0+3.0/x^5.0)
         end
     end
end

function moments(p0::ThetaMixturePrior2,μ,σ)
    η=p0.η
    α=μ/σ
    f=theta_mixt_factor2(α,η,p0.thr)
    χ=sqrt(2.0/π)*(2.0*η-1.0)*f
    av=μ+σ*χ
    va=σ^2.0*(1-χ^2.0)-μ*σ*χ
    return av,va
end
###############

function moments(p0::ThetaMixturePrior,μ,σ)
    η=p0.η
    α=μ/σ
    f=theta_mixt_factor(α,η)
    χ=sqrt(2.0/π)*(2.0*η-1.0)*f
    av=μ+σ*χ
    va=σ^2.0*(1-χ^2.0)-μ*σ*χ
    return av,va
end

function gradient(p0::ThetaMixturePrior,μ,σ,∂𝐹::Union{FreeEnGrad,Nothing})
    η=p0.η
    x=μ/σ/sqrt(2)
    num=2*erf(x)
    den=η*erfc(-x)+(1-η)*erfc(x)
    p0.η+=p0.δη*num/den
    p0.η=clamp(p0.η,0,1)
end

"""
Bernoulli-Bernoulli RBM
"""
struct RBM_Bias_Factor{T<:Real} <: Prior
    g::T
end

function moments(p0::RBM_Bias_Factor,μ,σ)
    arg=-p0.g+(1.0-2.0*μ)/(2.0*σ^2.0)
    av=1.0/(1.0+exp(arg))
    va=0.5/(1.0+cosh(arg))
    return av,va
end

"""
Gaussian RBM units
"""
struct RBM_Gaussian_Factor{T<:Real} <: Prior
    γ::T
    θ::T
end

function moments(p0::RBM_Gaussian_Factor,μ,σ)
    av = (μ+p0.θ*σ^2.0)/(1.0+p0.γ*σ^2.0)
    secmom = ((μ+p0.θ*σ^2.0)^2.0+σ^2.0*(1.0+p0.γ*σ^2.0))/(1+p0.γ*σ^2.0)^2.0
    #secmom = (μ^2.0+σ^2.0+2.0*p0.θ*μ*σ^2.0+(p0.γ+p0.θ^2.0)*σ^2.0)/(1+p0.γ*σ^2.0)^2.0
    #va = clamp(secmom-av^2.0,1e-50,1e50)
    va=1.0/(1.0/σ^2.0+p0.γ)
    return av,va
end
