"""
Code extracted from: Lattices and Coxeter Diagrams DZG.ipynb
This file contains only the code cells from the notebook.
"""

# === Code Cell 1 ===

from IPython.display import Math
import numpy as np
import pandas as pd
from IPython.display import HTML
from collections import Counter
from sage.modules.free_module_integer import IntegerLattice

H = IntegralLattice("H")
E8 = IntegralLattice("E8").twist(-1)
E82 = E8.twist(2)
H2 = H.twist(2)

def namestr(obj):
    namespace = globals()
    return [name for name in namespace if namespace[name] is obj][0]

S22 = SymmetricGroup(22)
rho = S22("(5, 9, 13, 1)(6, 10, 14, 2)(7, 11, 15, 3)(8, 12, 16, 4)(18, 19, 20, 17)(21, 22)")
s = S22("(1, 9)(2, 8)(3, 7)(4, 6)(10, 16)(11, 15)(12, 14)(17, 19)")
r = rho * rho
d = s
h = rho * s
v = s * rho
display(v)


# === Code Cell 2 ===

# Build (20, 2, 0) = U+U(2)+E_8+E_8, which contains (18, 2, 0) = U + E_8 + E_8 for the square Coxeter diagram

L_20_2_0 = H.direct_sum(H2).direct_sum(E8).direct_sum(E8)

dot = lambda x,y : x * L_20_2_0.gram_matrix() * y
nm = lambda x: dot(x, x)

Gram_L_20_2_0 = L_20_2_0.gram_matrix()
Gram_L_20_2_0.subdivide([2, 4, 12], [2, 4, 12])
L_20_2_0_dual_changeofbasis = Gram_L_20_2_0.inverse()
L_20_2_0_dual_changeofbasis.subdivide([2, 4, 12], [2, 4, 12])

e,f ,ep,fp, a1,a2,a3,a4,a5,a6,a7,a8, a1t,a2t,a3t,a4t,a5t,a6t,a7t,a8t = L_20_2_0.basis()
eb,fb, epb,fpb, w1,w2,w3,w4,w5,w6,w7,w8, w1t,w2t,w3t,w4t,w5t,w6t,w7t,w8t = L_20_2_0_dual_changeofbasis.columns()

# display(Math("$(18, 2, 0)="), Gram_L_20_2_0)
# display(Math("$(18, 2, 0)^{\\vee} =$"), L_20_2_0_dual_changeofbasis)

# The primes are the image of the diagonal embedding from E_8(2)
a1p = a1 + a1t
a2p = a2 + a2t
a3p = a3 + a3t
a4p = a4 + a4t 
a5p = a5 + a5t
a6p = a6 + a6t
a7p = a7 + a7t
a8p = a8 + a8t

w1p = w1 + w1t
w2p = w2 + w2t
w3p = w3 + w3t
w4p = w4 + w4t
w5p = w5 + w5t
w6p = w6 + w6t
w7p = w7 + w7t
w8p = w8 + w8t

MS      = [e,f,   ep,fp,   a1,a2,a3,a4,a5,a6,a7,a8, a1t,a2t,a3t,a4t,a5t,a6t,a7t,a8t]
MS_dual = [eb,fb, epb,fpb, w1,w2,w3,w4,w5,w6,w7,w8, w1t,w2t,w3t,w4t,w5t,w6t,w7t,w8t]

VS      = [ep,fp,   a1,a2,a3,a4,a5,a6,a7,a8, a1t,a2t,a3t,a4t,a5t,a6t,a7t,a8t]
VS_dual = [epb,fpb, w1,w2,w3,w4,w5,w6,w7,w8, w1t,w2t,w3t,w4t,w5t,w6t,w7t,w8t]

WS      = [e,f,   a1,a2,a3,a4,a5,a6,a7,a8, a1t,a2t,a3t,a4t,a5t,a6t,a7t,a8t]
WS_dual = [eb,fb, w1,w2,w3,w4,w5,w6,w7,w8, w1t,w2t,w3t,w4t,w5t,w6t,w7t,w8t]

for i, v in enumerate(VS):
    display(Math(f'V_{{{i+1}}}=' + str(v)))
for i, v in enumerate(VS_dual):
    display(Math(f'V_{{{i+1}}}^*=' + str(v)))

for i, v in enumerate(WS):
    display(Math(f'W_{{{i+1}}}=' + str(v)))
for i, v in enumerate(WS_dual):
    display(Math(f'W_{{{i+1}}}^*=' + str(v)))


# === Code Cell 3 ===

## Build (18, 0, 0) = U+E_8+E_8.

# L_18_0_0 = H.direct_sum(E8).direct_sum(E8)
# dot_w = lambda x,y : x * L_18_0_0.gram_matrix() * y
# nm_2 = lambda x: dot(x, x)


# Gram_L_18_0_0 = L_18_0_0.gram_matrix()
# Gram_L_18_0_0.subdivide([2, 10], [2, 10])

# display(Math("$(18, 0, 0)=$"), Gram_L_18_0_0)
# e_w,f_w,a1_w,a2_w,a3_w,a4_w,a5_w,a6_w,a7_w,a8_w,a1p_w,a2p_w,a3p_w,a4p_w,a5p_w,a6p_w,a7p_w,a8p_w = L_18_0_0.basis()
# WS = [e_w,f_w,a1_w,a2_w,a3_w,a4_w,a5_w,a6_w,a7_w,a8_w,a1p_w,a2p_w,a3p_w,a4p_w,a5p_w,a6p_w,a7p_w,a8p_w]

# L_18_0_0_dual_changeofbasis = Gram_L_18_0_0.inverse()
# L_18_0_0_dual_changeofbasis.subdivide([2, 10], [2, 10])

# display(Math("$(18, 0, 0)^{\\vee}=$"), L_18_0_0_dual_changeofbasis)

# eb_1800, fb_1800, w1_1800, w2_1800, w3_1800, w4_1800, w5_1800, w6_1800, w7_1800, w8_1800, w1_t_1800, w2_t_1800, w3_t_1800, w4_t_1800, w5_t_1800, w6_t_1800, w7_t_1800, w8_t_1800 = L_18_0_0_dual_changeofbasis.columns()
# WS_dual = [eb_1800, fb_1800, w1_1800, w2_1800, w3_1800, w4_1800, w5_1800, w6_1800, w7_1800, w8_1800, w1_t_1800, w2_t_1800, w3_t_1800, w4_t_1800, w5_t_1800, w6_t_1800, w7_t_1800, w8_t_1800]


# for i, w in enumerate(WS):
#     display(Math(f'$W_{{{i+1}}}=' + str(w)))
    
# for i, w in enumerate(WS_dual):
#     display(Math(f'$\\overline W_{{{i+1}}}=' + str(w)))


# === Code Cell 4 ===

# Starting new indexing
# l = [(i+1, i) for i in range(22) ]
# d = dict(l)
# H = PermutationGroup([[d[i] for i in g.tuple()] for g in S22.gens()], domain=d.values() )
# rho = H("(4,8,12,0)(5,9,13,1)(6,10,14,2)(7,11,15,3)(17,18,19,16)(20,21)")
# s = H("(0, 8)(1, 7)(2, 6)(3, 5)(9, 15)(10, 14)(11, 13)(16, 18)")
# r = rho * rho
# d = s
# h = rho * s
# v = s * rho


# === Code Cell 5 ===

# Checking lattice invariants:

# Check lattice invariants
# LS2 = IntegralLattice(MS3)
# LS2p = IntegerLattice(MS3)

# display("Lattice L^+ for Sterk 3:")
# display( "Signature: " + str( LS2.signature_pair()) )
# display( "Is an even lattice? " + str( LS2.is_even() ) )
# display( "Is unimodular? " + str(LS2p.is_unimodular() ) )

# invs = lattice_invariants( LS2 )
# display( "Attempt to compute lattice invariants for L^+:" ) 
# display(Math( "L^+" + ": (r,a,\\delta) = " + latex(invs["RAD"]) ))
# display(invs)

# LS2.discriminant_group()


# === Code Cell 6 ===

# Build a Coxeter diagram from a Coxeter matrix

def Coxeter_Diagram(M):
    nverts = M.ncols()
    # print(str(nverts) + " vertices")
    G = Graph()
    vertex_labels = dict();
# plot_coxeter_diagram(G)
    
    vertex_colors = {
        '#F8F9FE': [], # white
        '#BFC9CA': [], # black
    }
    
    for i in range(nverts):
        for j in range(nverts):
            mij = M[i, j]
            if i == j: 
                if mij == -2:
                    vertex_colors["#F8F9FE"].append(i) # white
                    continue
                if mij == -4:
                    vertex_colors["#BFC9CA"].append(i) # black
                    continue
                continue
            if mij > 0:
                G.add_edge(i, j, str(mij) )
                continue
    assert len( vertex_colors["#F8F9FE"]) + len( vertex_colors["#BFC9CA"]) == nverts
    G.vertex_colors = vertex_colors    
    return G

def plot_coxeter_diagram(G, v_labels, pos={}):
    n = len( G.vertices() )
    vlabs = {v: k for v, k in enumerate(v_labels)}
    if pos == {}:
        display(G.plot(
            edge_labels=True, 
            vertex_labels = vlabs,
            vertex_size=200,
            vertex_colors = G.vertex_colors
        ))
    else:
        display(G.plot(
            edge_labels=True, 
            vertex_labels = vlabs,
            vertex_size=200,
            vertex_colors = G.vertex_colors,
            pos = pos
        ))
        
def root_intersection_matrix(vectors, labels, bil_form):
    n = len(vectors)
    M = zero_matrix(ZZ, n)
    nums = Set(range(n))
    for i in range(n):
        for j in range(n):
            M[i, j] = bil_form( vectors[i], vectors[j] )

    print("Diagonal entries/square norms: ")
    display(M.diagonal())

    # Labels!
    
    
    df = pd.DataFrame(M, columns=labels, index=labels)
    display(HTML(df.to_html()))
            
    # Must be symmetric
    assert M.is_symmetric()
        
    # Must have -2 or -4 on the diagonal
    s = Set( M.diagonal() )
    assert s in Subsets( Set( [-2, -4] ) )

    # Diagonals should be square norms of vectors
    for i in range(n):
        assert M[i, i] == bil_form(vectors[i], vectors[i])

    

    return M

def lattice_invariants(L):
    G = L.gram_matrix()
    A = L.discriminant_group()
    Q = A.gram_matrix_quadratic()
    D,U,V = Q.smith_form()
    
    if len(list(set(Q.diagonal()) - set([1]) )) >= 2:
        print("Not a p-elementary lattice. Multiple distinct elementary divisors: ", Q.diagonal())
    
    diagonal_all_integers = reduce(lambda x,y : x and y, [s.is_integer() for s in Q.diagonal()] )
    delta = 0 if diagonal_all_integers else 1    
    
    return {
        "RAD": (L.rank(), len(Q.elementary_divisors()), delta),
        "elementary divisors": Q.elementary_divisors(),
        "SNF": D, 
        "discriminant form": Q,        
        "discriminant group": A
    }

    
# Counter( MS3.elementary_divisors() )
# D,U,V = MS3.smith_form()
# display(D)
# display(MS3.elementary_divisors() )
# from collections import Counter
# Counter(MS3.elementary_divisors() )
# Test
#M = Matrix(ZZ, 4, [ [-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 2], [0, 0, 2, -4] ])
#display(M)
#G = Coxeter_Diagram(M)
#plot_coxeter_diagram(G, v_labels = [f"$r_{ {i + 1} }$" for i in range( 4 )] )


# === Code Cell 7 ===

# display( lattice_invariants(H.direct_sum(H.twist(2)).direct_sum(E8).direct_sum(E8) ) )
# display( lattice_invariants(H.direct_sum(H.twist(2)).direct_sum(E8.twist(2)) ) )


# === Code Cell 8 ===

# Root vectors for (18, 2, 0), roots taken from above, v_i are according to numerical labeling above

v1 = a8t
v2 = ep + fp + w1 + w8t
v3 = a1 
v4 = a3
v5 = a4
v6 = a5
v7 = a6
v8 = a7
v9 = a8
v10 = ep + fp + w8 + w1t
v11 = a1t
v12 = a3t
v13 = a4t
v14 = a5t
v15 = a6t
v16 = a7t

v17 = ep + w8t
v18 = a2
v19 = ep + w8
v20 = a2t

v21 = fp - ep
v22 = 5 ep + 3 fp + 2 w2 + 2 w2t

V = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22]
MV = root_intersection_matrix(V, labels = [f"$v_{ {r + 1} }$" for r in range( len(V) )], bil_form=dot)
# for v in V:
#     display(v)


# === Code Cell 9 ===

dot(v22, v22)


# === Code Cell 10 ===

G = Coxeter_Diagram(MV)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$v_{ {i + 1} }$" for i in range( 22 )],
    pos = {
        0: [0, 0],
        1: [4, 0],
        2: [8, 0],
        3: [12, 0],
        4: [16, 0],
        5: [16, -4],
        6: [16, -8],
        7: [16, -12],
        8: [16, -16],
        9: [12, -16],
        10: [8, -16],
        11: [4, -16],
        12: [0, -16],
        13: [0, -12],
        14: [0, -8],
        15: [0, -4],
        16: [4, -4],
        17: [12, -4],
        18: [12, -12],
        19: [4, -12],
        20: [8, -10],
        21: [8, -6],
    }
)


# === Code Cell 11 ===

# Root vectors for (18, 0, 0), roots taken from above, w_i are according to numerical labeling above

w1 = a1
w2 = a3
w3 = a4
w4 = a5
w5 = a6
w6 = a7
w7 = a8
w8 = w8 + e
w9 = f- e
w10 = w8t + e
w11 = a8t
w12 = a7t
w13 = a6t
w14 = a5t
w15 = a4t
w16 = a3t
w17 = a1t
w18 = a2
w19 = a2t

W = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19]
MW = root_intersection_matrix(W, labels = [f"$w_{ {r + 1} }$" for r in range( len(W) )], bil_form=dot)


G = Coxeter_Diagram(MW)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$w_{ {i + 1} }$" for i in range( 19 )],
    pos = {
        0: [-4, 0],
        1: [-8, 0],
        2: [-12, 0],
        3: [-10, 4],
        4: [-8, 8],
        5: [-6, 12],
        6: [-4, 16],
        7: [-2, 20],
        8: [0, 24],
        9: [2, 20],
        10: [4, 16],
        11: [6, 12],
        12: [8, 8],
        13: [10, 4],
        14: [12, 0],
        15: [8, 0],
        16: [4, 0],
        17: [-4, 4],
        18: [4, 4]
    }
)


# === Code Cell 12 ===

# Sterk 1

display(r.cycle_tuples(singletons=True))

s1_1 = v3 + v11
s1_2 = v4 + v12
s1_3 = v5 + v13
s1_4 = v6 + v14
s1_5 = v7 + v15
s1_6 = v8 + v16
s1_7 = v9 + v1
s1_8 = v10 + v2
s1_9 = v17 + v19
s1_10 = v21
s1_11 = v22
s1_12 = v18 + v20

S1 = [s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9, s1_10, s1_11, s1_12]
MS1 = root_intersection_matrix(S1, labels = [f"$s^1_{ {r + 1} }$" for r in range( len(S1) )], bil_form=dot)

G = Coxeter_Diagram(MS1)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$s^1_{ {i + 1} }$" for i in range( 22 )],
    pos = {
        0: [0, 0],
        1: [4, 0],
        2: [8, 0],
        3: [8, -4],
        4: [8, -8],
        5: [4, -8],
        6: [0, -8],
        7: [0, -4],
        8: [0, -8],
        9: [2, -6],
        10: [4, -4],
        11: [6, -2]
    }
)


# === Code Cell 13 ===

S1roots = [
    a1p, #1
    a2p, #2
    a3p, #3
    a4p, #4
    a5p, #5
    a6p, #6
    a7p, #7
    a8p, #8
    fp-ep, #9
    w8p + 2ep, #10
    2ep + 2fp + w1p + w8p, #11
    5ep + 3fp + 2w2p #12
]

labs = [f"$S^1_{ {r + 1} }$" for r in range( len(S1roots) )]

plot_coxeter_diagram(
    Coxeter_Diagram(
        root_intersection_matrix(
            S1roots, 
            labels = labs, 
            bil_form=dot
        )
    ), 
    v_labels = labs,
    pos = {
        0: [0, 0],
        1: [-4, -8],
        2: [-4, -4],
        3: [-8, -8],
        4: [-4, -12],
        5: [0, -16],
        6: [4, -12],
        7: [8, -8],
        8: [2, -8],
        9: [5, -8],
        10: [4, -4],
        11: [-1, -8]
}
)


# === Code Cell 14 ===

# Sterk 2

s2_1 = w1 + w17
s2_2 = w2 + w16
s2_3 = w3 + w15
s2_4 = w4 + w14
s2_5 = w5 + w13
s2_6 = w6 + w12
s2_7 = w7 + w11
s2_8 = w8 + w10
s2_9 = w9
s2_10 = w18 + w19

S2 = [s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9, s2_10]
MS2 = root_intersection_matrix(S2, labels = [f"$s^2_{ {r + 1} }$" for r in range( len(S2) )], bil_form=dot )

G = Coxeter_Diagram(MS2)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$s^2_{ {i + 1} }$" for i in range( 22 )],
    pos = {
        0: [0, 0],
        1: [-4, 0],
        2: [-8, 0],
        3: [-7, 4],
        4: [-6, 8],
        5: [-5, 12],
        6: [-4, 16],
        7: [-3, 20],
        8: [-2, 24],
        9: [-2, 6]
    }
)


# === Code Cell 15 ===

# Sterk's own basis for Sterk 2
S2roots = [
    a1p, #1
    a2p, #2
    a3p, #3
    a4p, #4
    a5p, #5
    a6p, #6
    a7p, #7
    a8p, #8
    w8p + 2f, #9
    e-f, #10
]

labs = [f"$S^2_{ {r + 1} }$" for r in range( len(S2roots) )]

plot_coxeter_diagram(
    Coxeter_Diagram(
        root_intersection_matrix(
            S2roots, 
            labels = labs, 
            bil_form=dot
        )
    ), 
    v_labels = labs,
    pos = {
        0: [0, 0],
        1: [8, -4],
        2: [4, 0],
        3: [8, 0],
        4: [12, 0],
        5: [16, 0],
        6: [20, 0],
        7: [24, 0],
        8: [28, 0],
        9: [32, 0]
}
)


# === Code Cell 16 ===

# Sterk 3

display(d.cycle_tuples(singletons=True))

wa = lambda x: x + (1/2)*dot(v22, x) * v22
I = lambda x: x + wa(x)

s3_1 = v13
s3_2 = v14 + v12
s3_3 = v15 + v11
s3_4 = v16 + v10
s3_5 = v1 + v9
s3_6 = v2 + v8
s3_7 = v3 + v7
s3_8 = v4 + v6
s3_9 = v5
s3_10 = v17 + v19
s3_11 = I(v20) #v22 + 2*v20
s3_12 = I(v18) #v22 + 2*v18

S3 = [s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8, s3_9, s3_10, s3_11, s3_12]
MS3 = root_intersection_matrix(S3, labels = [f"$s^2_{ {r + 1} }$" for r in range( len(S3) )], bil_form=dot )

G = Coxeter_Diagram(MS3)
pos_dict = {
        0: [0, -4],
        1: [0, 4],
        2: [0, 8],
        3: [0, 12],
        4: [0, 16],
        5: [4, 16],
        6: [8, 16],
        7: [12, 16],
        8: [20, 16],
        9: [4, 12],
        10: [6, 2],
        11: [14, 10],
    }
plot_coxeter_diagram(
    G, 
    v_labels = [f"$s^3_{ {i + 1} }$" for i in range( len(S3) )],
    pos = pos_dict
)

#pos_dict


# === Code Cell 17 ===

# # Sterk's own basis for Sterk 3
sa1 = a1p
sa2 = a2p
sa3 = a3p
sa4 = a4p
sa5 = a5p
sa6 = a6p
sa7 = a7p
# alpha_8 is not on this diagram
sa9 = f-e
sa10 = 2fp + w8p
sa11 = 2e - 2fp - w8p
sa12 = 2e + w1p - w8p
sa13 = e + f + a8p - fp

S3roots = [
    sa4, #1
    sa3, #2
    sa1, #3
    sa12, #4
    sa9, #5
    sa11, #6
    sa10, #7
    sa13, #8
    sa7, #9
    sa6, #10
    sa5, #11
    sa2 #12
]

labs = [f"$S^3_{ {r + 1} }$" for r in range( len(S3roots) )]

plot_coxeter_diagram(
    Coxeter_Diagram(
        root_intersection_matrix(
            S3roots, 
            labels = labs, 
            bil_form=dot
        )
    ), 
    v_labels = labs,
    pos = {
        0: [0, 0],
        1: [-4, -4],
        2: [-8, -8],
        3: [-12, -12],
        4: [-8, -16],
        5: [-4, -20],
        6: [4, -20],
        7: [8, -16],
        8: [12, -12],
        9: [8, -8],
        10: [4, -4],
        11: [0, -4]
    }
)


# === Code Cell 18 ===

# Sterk 4

# display(v.cycle_tuples(singletons=True))

s4_1 = v15
s4_2 = v16 + v14
s4_3 = v1 + v13
s4_4 = v2 + v12
s4_5 = v3 + v11
s4_6 = v4 + v10
s4_7 = v5 + v9
s4_8 = v6 + v8
s4_9 = v7
s4_10 = v17 + v20
s4_11 = v18 + v19
s4_12 = v22 + v21

# Although s412 is an invariant vector, it is not a root:
# Math('(s^4_{12})^2=' + str( nm(s4_12)))

S4 = [s4_1, s4_2, s4_3, s4_4, s4_5, s4_6, s4_7, s4_8, s4_9, s4_10, s4_11]
MS4 = root_intersection_matrix(S4, labels = [f"$s^4_{ {r + 1} }$" for r in range( len(S4) )], bil_form=dot)

G = Coxeter_Diagram(MS4)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$s^4_{ {i + 1} }$" for i in range( 11 )],
    pos = {
        0: [0, 0],
        1: [0, 4],
        2: [0, 8],
        3: [4, 8],
        4: [8, 8],
        5: [12, 8],
        6: [16, 8],
        7: [16, 4],
        8: [16, 0],
        9: [4, 4],
        10: [12, 4]    
}
)


# === Code Cell 19 ===

# Sterk's own roots for Sterk 4

# sa1 = a1p no alpha_1 on this diagram
sa2 = a2p
sa3 = a3p
sa4 = a4p
sa5 = a5p
sa6 = a6p
sa7 = a7p
sa8 = a8p
sa9 = f-e
sa10 = w8p + 2ep
sa11 = e + f + a1p - ep
sa12 = 2e - 2ep + w8p - w1p

S4roots = [
    sa11, #1
    sa3, #2
    sa4, #3
    sa5, #4
    sa6, #5
    sa7, #6
    sa8, #7
    sa12, #8
    sa9, #9
    sa2, #10
    sa10, #11
]

labs = [f"$S^4_{ {r + 1} }$" for r in range( len(S4roots) )]

plot_coxeter_diagram(
    Coxeter_Diagram(
        root_intersection_matrix(
            S4roots, 
            labels = labs, 
            bil_form=dot
        )
    ), 
    v_labels = labs,
    pos = {
        0: [0, 0],
        1: [4, 0],
        2: [8, 0],
        3: [12, 0],
        4: [16, 0],
        5: [20, 0],
        6: [24, 0],
        7: [28, 0],
        8: [32, 0],
        9: [8, -4],
        10: [24, -4]
    }
)


# === Code Cell 20 ===

# Sterk 5

s5_1 = v16 + 2v1 + v2
s5_2 = v2 + 2v3 + v4
s5_3 = v4 + 2v5 + v6
s5_4 = v6 + 2v7 + v8
s5_5 = v8 + 2v9 + v10
s5_6 = v10 + 2v11 + v12
s5_7 = v12 + 2v13 + v14
s5_8 = v14 + 2v15 + v16
s5_9 = v17
s5_10 = v18
s5_11 = v19
s5_12 = v20
s5_13 = v21
s5_14 = v22

S5 = [s5_1, s5_2, s5_3, s5_4, s5_5, s5_6, s5_7, s5_8, s5_9, s5_10, s5_11, s5_12, s5_13, s5_14]
MS5 = root_intersection_matrix(S5, labels = [f"$s^5_{ {r + 1} }$" for r in range( len(S5) )], bil_form=dot)

G = Coxeter_Diagram(MS5)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$s^5_{ {i + 1} }$" for i in range( 14 )],
    pos = {
        0: [0, 0],
        1: [10, 0],
        2: [20, 0],
        3: [20, -10],
        4: [20, -20],
        5: [10, -20],
        6: [0, -20],
        7: [0, -10],
        8: [4, -4],
        9: [16, -4],
        10: [16, -16],
        11: [4, -16],
        12: [8, -8],
        13: [8, -12]
}
)


# === Code Cell 21 ===

s_a1_tilde = ep - fp
s_a1_tildebar = epb - fpb
display("dot with a1 tilde bar")
show( dot(s_a1_tilde, s_a1_tilde_bar) )
# show( dot(s_a2_tilde, s_a2_tilde_bar) )
# show( dot(s_a3_tilde, s_a3_tilde_bar) )
# show( dot(s_a6_tilde, s_a6_tilde_bar) )
# show( dot(s_a8_tilde, s_a8_tilde_bar) )
def namestr(obj):
    namespace = globals()
    return [name for name in namespace if namespace[name] is obj][1]
    
for v in [e,f,   ep,fp,   a1,a2,a3,a4,a5,a6,a7,a8, a1t,a2t,a3t,a4t,a5t,a6t,a7t,a8t]:
    display("dot with " + namestr(v) )
    show( dot(s_a1_tilde_bar, v) )


# === Code Cell 22 ===

# Sterk's own roots for Sterk 5

s_e_tilde = e + ep + fp - a1p
s_f_tilde = f + ep + fp - a1p

s_a1_tilde = ep - fp
s_a2_tilde = a2p
s_a3_tilde = fp + a3p
s_a4_tilde = a4p
s_a5_tilde = a5p
s_a6_tilde = a6p
s_a7_tilde = a7p
s_a8_tilde = a8p

s_a1_tilde_bar = epb - fpb # probablt not
s_a2_tilde_bar = w2p
s_a3_tilde_bar = fpb + w3p # probably not
s_a4_tilde_bar = w5p
s_a5_tilde_bar = w5p
s_a6_tilde_bar = w6p
s_a7_tilde_bar = w7p
s_a8_tilde_bar = w8p


sa9  = 2 s_e_tilde - s_a1_tilde
sa10 = 2 s_e_tilde + s_a6_tilde_bar - s_a3_tilde_bar # probably not
sa11 = s_f_tilde - s_e_tilde
sa12 = s_e_tilde + s_f_tilde + s_a6_tilde_bar - s_a3_tilde_bar # probably not
sa13 = s_e_tilde + s_f_tilde + s_a1_tilde_bar + s_a8_tilde_bar - s_a3_tilde_bar # probably not
sa14 = s_e_tilde + s_f_tilde + s_a3_tilde


S5roots = [
    a4p, #1
    a2p, #2
    # sa10, #3, issue
    s_a8_tilde_bar, #4
    a8p, #5
    a7p, #6
    a6p, #7
    a5p, #8
    sa14, #9
    s_a1_tilde, #10
    # sa13, #11, issue
    # sa12, #12, issue
    sa9, #13,
    sa11, #14
]

labs = [f"$S^5_{ {r + 1} }$" for r in range( len(S5roots) )]

plot_coxeter_diagram(
    Coxeter_Diagram(
        root_intersection_matrix(
            S5roots, 
            labels = labs, 
            bil_form=dot
        )
    ), 
    v_labels = labs,
    pos = {
        0: [0, 0],
        1: [4, 4],
        2: [8, -4],
        3: [12, 4],
        4: [16, -4],
        5: [20, 4],
        6: [24, -4],
        7: [28, 4],
        8: [32, -4],
        9: [36, 4],
        10: [40,-4],
        11: [44, 4], 
        12: [48, -4],
        13: [52, 4]
    }
)


# === Code Cell 23 ===

for i, s in enumerate(S5roots):
    show(i+1, ":   ", s)


# === Code Cell 24 ===

sa10


# === Code Cell 25 ===

Gram_L_20_2_0


# === Code Cell 26 ===

cb1 = vector( [1,0,1,1,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0] )
cb2 = vector( [0,1,1,1,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0] )
cb5 = vector( [0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] )
cb7 = vector( [0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] )
cbz = vector( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] )

# display( cb1 * Gram_L_20_2_0 * cb1 )
# display( cb2 * Gram_L_20_2_0 * cb2 )
# display( cb5 * Gram_L_20_2_0 * cb5 )
# display( cb7 * Gram_L_20_2_0 * cb7 )

COB = identity_matrix(20)

COB[:, 1] = cb1
# COB[:, 2] = cb2
# COB[:, 5] = cb5
# COB[:, 7] = cb7

det(COB)
display(COB)

# M = Matrix(ZZ, 20, [cb1, cb2, cbz, cbz, cb5, cbz, cb7, cbz, cbz, cbz, cbz, cbz, cbz, cbz, cbz, cbz, cbz, cbz, cbz, cbz, ])


# === Code Cell 27 ===

v5 = L_20_2_0(2e + 2f + w1 + w1t)
v5


# === Code Cell 28 ===

matrix(ZZ, 2, [0, 1, 1, 0]).inverse()


# === Code Cell 29 ===

dot(e, (1/2) f)


# === Code Cell 30 ===

show( eb == f )
show (fb == e )
show( eb - fb == f-e )


# === Code Cell 31 ===

# Lw = H.direct_sum(H2).direct_sum(E8.twist(2))
# b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12 = Lw.gens()
# Lw.gram_matrix()


# === Code Cell 32 ===

# B = identity_matrix(12)
# B[:, 0] = b1 + b3 + b4 - b5
# B[:, 1] = b2 + b3 + b4 - b5
# # B[:, 4] = b3 - b4
# B[:, 6] = b4 + b7
# B


# === Code Cell 33 ===

B.det()


# === Code Cell 34 ===

# L = H.direct_sum(H2).direct_sum(E8.twist(2))
# GL = L.gram_matrix()
# GL


# === Code Cell 35 ===

e, f, ep, fp, a1, a2, a3, a4, a5, a6, a7, a8 = L.gens()


# === Code Cell 36 ===

# ed, fd, epd, fpd, w1, w2, w3, w4, w5, w6, w7, w8 = GL.inverse().columns()
# ZS = [ed, fd, epd, fpd, w1, w2, w3, w4, w5, w6, w7, w8]
# for z in ZS:
#     show(z)


# === Code Cell 37 ===

w8t


# === Code Cell 38 ===

b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12 =  L.gens()
B = identity_matrix(ZZ, 12)
B[:, 0] = b1 + b3 + b4 - b5
B[:, 1] = b2 + b3 + b4 - b5
B[:, 4] = b3 - b4
B[:, 6] = b4 + b7


# === Code Cell 39 ===

B


# === Code Cell 40 ===

det(B)


# === Code Cell 41 ===

e_st = e + ep + fp - a1p
f_st = f + ep + fp - a1p
a1_st = ep - fp
a2_st = a2p
a3_st = fp + a3p
a4_st = a4p
a5_st = a5p
a6_st = a6p
a7_st = a7p
a8_st = a8p

ST = [e_st, f_st, a1_st, a2_st, a3_st, a4_st, a5_st, a6_st, a7_st, a8_st]

v_intersects = [dot(v, x) for x in ST]
v_intersects


# === Code Cell 42 ===

M.rank()


# === Code Cell 43 ===

labels = [f"$K_{ {r + 1} }$" for r in range( len(ST) )]
n = len(ST)
M = zero_matrix(ZZ, n)
nums = Set(range(n))
for i in range(n):
    for j in range(n):
        M[i, j] = dot( ST[i], ST[j] )

print("Diagonal entries/square norms: ")
display(M.diagonal())

# Labels!


df = pd.DataFrame(M, columns=labels, index=labels)
display(HTML(df.to_html()))


# === Code Cell 44 ===

# H.direct_sum(E8.twist(2)).gram_matrix()


# === Code Cell 45 ===

v = 2e + 2f + w1


# === Code Cell 46 ===

for s in ST:
    show(s)
    show( dot(v, s) )


# === Code Cell 47 ===

Ld = L.dual_lattice()
Ld.gram_matrix()


# === Code Cell 48 ===

len(a1_st)


# === Code Cell 49 ===

L_20_2_0


# === Code Cell 50 ===

L_20_2_0_dual = L_20_2_0.dual_lattice()
L_20_2_0_dual


# === Code Cell 51 ===

fpd = L_20_2_0_dual(fp)
fpd


# === Code Cell 52 ===

fp


# === Code Cell 53 ===

dot(fp, fpd)


# === Code Cell 54 ===

E8.gram_matrix().inverse()


# === Code Cell 55 ===

v = 2e + 2e + w1p


# === Code Cell 56 ===

v


# === Code Cell 57 ===

etilde = e + ep + fp - a1p
etilde


# === Code Cell 58 ===

dot(v, etilde)


# === Code Cell 59 ===

ftilde = f + ep + fp - a1p
dot(v, ftilde)


# === Code Cell 60 ===

def root_intersection_matrix(vectors, labels, bil_form):
    n = len(vectors)
    M = zero_matrix(ZZ, n)
    nums = Set(range(n))
    for i in range(n):
        for j in range(n):
            M[i, j] = bil_form( vectors[i], vectors[j] )

    # Must be symmetric
    assert M.is_symmetric()

    # Must have -2 or -4 on the diagonal
    s = Set( M.diagonal() )
    assert s in Subsets( Set( [-2, -4] ) )

    # Diagonals should be square norms of vectors
    for i in range(n):
        assert M[i, i] == bil_form(vectors[i], vectors[i])
    return M
    
def pp_root_matrix(M, labels, bil_form):
    print("Diagonal entries/square norms: ")
    display(M.diagonal())
    
    df = pd.DataFrame(M, columns=labels, index=labels)
    display(HTML(df.to_html()))


# === Code Cell 61 ===

Ws_parab_one = [w1, w2, w3, w4, w5, w6, w7, w8, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19]

M_Ws_parab_one = root_intersection_matrix(Ws_parab_one, labels = [f"$w_{ {r + 1} }$" for r in [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19] ], bil_form=dot)


G = Coxeter_Diagram(M_Ws_parab_one)
plot_coxeter_diagram(
    G, 
    v_labels = [f"$w_{ {i + 1} }$" for i in range( 19 )],
    pos = {
        0: [-4, 0],
        1: [-8, 0],
        2: [-12, 0],
        3: [-10, 4],
        4: [-8, 8],
        5: [-6, 12],
        6: [-4, 16],
        7: [-2, 20],
        8: [2, 20],
        9: [4, 16],
        10: [6, 12],
        11: [8, 8],
        12: [10, 4],
        13: [12, 0],
        14: [8, 0],
        15: [4, 0],
        16: [-4, 4],
        17: [4, 4]
    }
)


# === Code Cell 62 ===

print(f"Subdiagram is parabolic? {is_parabolic_matrix(M_Ws_parab_one)}")

print(f"Number of vertices in full diagram: {len(W)}")
print(f"Number of vertices in subdiagram: {len(Ws_parab_one)}")
print(f"Rank of full lattice: {matrix(ZZ, W).rank()}")
print(f"Rank of lattice generated by vectors in this subdiagram: {matrix(ZZ, Ws_parab_one).rank() }")


# === Code Cell 63 ===

W = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19]
MW = root_intersection_matrix(W, labels = [f"$w_{ {r + 1} }$" for r in range( len(W) )], bil_form=dot)

G = Coxeter_Diagram(MW)

verts = set(list( G.vertices()) )

all_subs_verts = subsets(verts)

small_subgraph_sample = [l for l in all_subs_verts if len(l) >= 16]

print(f"Number of subgraphs: {len(small_subgraph_sample)}")


# === Code Cell 64 ===

subgraphs = [G.subgraph(vs) for vs in small_subgraph_sample ]
test_case = [H for H in subgraphs if len(H.vertices()) >= 1 ]

full_rank = matrix(ZZ, W).rank()

print(f"Test case: {len(test_case)} subgraphs.")
print(f"Full matrix rank = {full_rank} and number of vertices = {len(G.vertices()) }")

# Seems we need to reverse the convention for parabolic/elliptic to be negative (semi)definite.

def is_elliptic_matrix(M):
    return (-1 * M).is_positive_definite()

def is_parabolic_matrix(M):
    return (-1 * M).is_positive_semidefinite()

def is_maximal_parabolic(M):
    return is_parabolic_matrix(M) and true;

def roots_from_subgraph(H):
    return [V[index] for index in H.vertices()]


# === Code Cell 65 ===

for H in test_case:
   
    # show(H)
    cpts = H.connected_components()
    num_cpts = len( cpts)
    H_verts = H.vertices()

    H_roots = [V[index] for index in H.vertices()]
    M1 = root_intersection_matrix(H_roots, labels = H_verts, bil_form=dot)
    rk = matrix(ZZ, H_roots).rank()
    order = len(H.vertices())
    if rk < 16:
        continue
    
    # pp_root_matrix(M1, labels = H_verts, bil_form=dot)
    if is_elliptic_matrix(M1):
        # print(f"Elliptic of rank {rk} (Type III)")
        # print(f"Vertices: {H_verts}")
        print("----------------------------------")
        continue
    elif is_parabolic_matrix(M1):
        if rk >= 17:
            print(f"Parabolic of rank {rk} (Type II)")
            print(H_verts)
            show(H)
        else:
            print("Non-maximal parabolic found")
            continue
    else:
        # print(f"Neither elliptic nor parabolic; rank {rk}")
        continue
    show(M1)


# === Code Cell 66 ===

[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18] in small_subgraph_sample

