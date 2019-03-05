export Factor


abstract type Factor end


"""
    moments!(av, va, p0::T, h, J) where T <:Factor -> (mean, variance)

    input: ``p_0, h, J``

    output: mean and variance of

    `` p(x) ∝ p_0(x) exp(-½⋅J⋅x² + h⋅x)``
"""
function moments!(av, va, p0::T, h, J) where T <: Factor
    error("undefined moment calculation, assuming uniform prior")
    return J\h,J\I
end

"""

    learn!(p0::T, h, J) -> nothing

    update parameters with a single learning gradient step (learning rate is stored in p0)
"""
function learn!(p0::T, h, J) where T <: Factor
    #by default, do nothing
    return
end

