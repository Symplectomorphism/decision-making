using Base.Iterators
using Distributions
using LinearAlgebra
using Graphs

struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    𝖳  # transition function
    𝖱  # reward function
    𝖳𝖱 # sample transition and reward
end
