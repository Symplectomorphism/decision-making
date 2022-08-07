using Base.Iterators
using Distributions
using LinearAlgebra

include("utility.jl")

struct Variable
    name::Symbol
    m::Int # number of possible values
end

const Assignment = Dict{Symbol, Int}
const FactorTable = Dict{Assignment, Float64}

struct Factor
    vars::Vector{Variable}
    table::FactorTable
end

variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]

select(a::Assignment, varnames::Vector{Symbol}) = 
    Assignment(n=>a[n] for n in varnames)


function assignments(vars::AbstractVector{Variable})
    names = [var.name for var in vars]
    return vec(
        [Assignment(n=>v for (n,v) in zip(names, values))
         for values in product((1:v.m for v in vars)...)]
    )
end

function normalize!(ϕ::Factor)
    z = sum(p for (a,p) in ϕ.table)
    for (a,p) in ϕ.table
        ϕ.table[a] = p/z
    end
    return ϕ
end
