include("inference.jl")

B = Variable(:b, 2); S = Variable(:s, 2)
E = Variable(:e, 2)
DA = Variable(:da, 2); DB = Variable(:db, 2); 
C = Variable(:c, 2)
vars = [B, S, E, DA, DB, C]
factors = [
    Factor([B], FactorTable((b=1,) => 0.99, (b=2,) => 0.01)),
    Factor([S], FactorTable((s=1,) => 0.98, (s=2,) => 0.02)),
    Factor([E,B,S], FactorTable(
        (e=1,b=1,s=1) => 0.90, (e=1,b=1,s=2) => 0.04,
        (e=1,b=2,s=1) => 0.05, (e=1,b=2,s=2) => 0.01,
        (e=2,b=1,s=1) => 0.10, (e=2,b=1,s=2) => 0.96,
        (e=2,b=2,s=1) => 0.95, (e=2,b=2,s=2) => 0.99)),
    Factor([DA,E], FactorTable(
        (da=1,e=1) => 0.96, (da=1,e=2) => 0.03,
        (da=2,e=1) => 0.04, (da=2,e=2) => 0.97)),
    Factor([DB,E], FactorTable(
        (db=1,e=1) => 0.96, (db=1,e=2) => 0.03,
        (db=2,e=1) => 0.04, (db=2,e=2) => 0.97)),
    Factor([C,E], FactorTable(
        (c=1,e=1) => 0.98, (c=1,e=2) => 0.01,
        (c=2,e=1) => 0.02, (c=2,e=2) => 0.99))
]
graph = SimpleDiGraph(6)
add_edge!(graph, 1, 3); add_edge!(graph, 2, 3)
add_edge!(graph, 3, 4); add_edge!(graph, 3, 5); 
add_edge!(graph, 3, 6)
bn = BayesianNetwork(vars, factors, graph)

# fine = (b=1, s=1, e=1, d=1, c=1)
# fine_dev = (b=1, s=1, e=1, d=2, c=1)
# display( probability(bn, Assignment(fine_dev)) )

M = ExactInference()
V = VariableElimination([3,4,5,6,2,1])
D = DirectSampling(10_000)
G = GibbsSampling(10_000, 1_000, 100, topological_sort_book(bn.graph))
evidence = (da=2, db=2)
pre = reduce(marginalize, (prod(bn.factors), :e, :c, :da, :db))
post = infer(M, bn, [:b, :s], evidence)
postvel = infer(V, bn, [:b, :s], evidence)
postsamp = infer(D, bn, [:b, :s], evidence)
postGibbs = infer(G, bn, [:b, :s], Dict(pairs(evidence)))

display(pre.table)
display(post.table)
display(postvel.table)
display(postsamp.table)
display(postGibbs.table)
