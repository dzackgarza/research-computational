"""
Code extracted from: Curve Intersection Multiplicities.ipynb
This file contains only the code cells from the notebook.
"""

# === Code Cell 1 ===

def intersection_number(F, G, point = (0,0)):
    (x,y) = determine_variables(F, G)

    # translate both curves to origin and calculate it there
    F = F.subs({x:x + point[0], y:y + point[1]})
    G = G.subs({x:x + point[0], y:y + point[1]})

    # if $F(0,0)\neq 0$ or $G(0,0)\neq 0$ they don't intersect in the origin
    if F.subs({x:0, y:0}) != 0 or G.subs({x:0, y:0}) != 0: return 0

    # if $F$ or $G$ are zero they don't intersect properly
    if F == 0 or G == 0: return Infinity

    # we only look at factors of $x$
    f = F.subs({y:0})
    g = G.subs({y:0})

    # $F$ contains a component $y=0$
    if f == 0:
        # $G$ contains a component $y=0$ too, no proper intersection
        if g == 0: return infinity
        # remove common $y^n$ in $F$, count degree of $x$ in $G$ and recurse
        else:
            f = F.quo_rem(y)[0]
            return ldegree(g) + intersection_number(f, G)
    # $F$ does not contain a component $y=0$
    else:
        # $G$ *does* contain a component $y=0$
        if g == 0:
            g = G.quo_rem(y)[0]
            return ldegree(f) + intersection_number(F, g)
        # we recurse as in condition (7) by removing factors of $x$
        else:
            a, b = f.lc(), g.lc()
            r, s = f.degree(), g.degree()

            # we drop the highest degree term
            if r <= s:
                return intersection_number(F, a*G - b*x^(s-r)*F)
            else:
                return intersection_number(b*F - a*x^(r-s)*G, G)

def ldegree(F):
    minimum = infinity

    for (n, m) in F.dict():
        minimum = min(minimum, n)

    return minimum

def determine_variables(F, G):
    if len(F.variables()) == 2:
        return F.variables()

    if len(G.variables()) == 2:
        return G.variables()

    if len(F.variables()) == len(G.variables()) == 1:
        if F.variables() == G.variables():
            return (F.variable(0), 0)
        else:
            return (G.variable(0), F.variable(0))

    return (0,0)


# === Code Cell 2 ===

P.<x,y> = PolynomialRing(QQ, 2)
F = []
F.append(y-x^2);
F.append(y^2-x^3+x);
F.append(y^2-x^3);
F.append(y^2-x^3-x^2);
F.append((x^2+y^2)^2+3*x^2*y-y^3);
F.append((x^2+y^2)^3-4*x^2*y^2);

for i in range(0,6):
    for j in range(i + 1,6):
        print('(%(i)d, %(j)d): %(m)d' % {'i': i, 'j': j, 'm': intersection_number(F[i], F[j])} )


# === Code Cell 3 ===

intersection_number( x+y^2, x+y^2-x^3)


# === Code Cell 4 ===

A.<x,y> = AffineSpace(CC, 2)
X = A.subscheme([x+y^2])
Y = A.subscheme([x+y^2-x^3])
Q1 = Y([0,0])
Q1.intersection_multiplicity(X)


# === Code Cell 5 ===

A.<x,y,z> = AffineSpace(CC, 3)
X = A.subscheme([y^2 - x^7*z])
Q1 = X([1,1,1])
Q1.multiplicity()


# === Code Cell 6 ===

Q2 = X([0,0,2])
Q2.multiplicity()


# === Code Cell 7 ===

# Note singularities all along z-axis: https://www.desmos.com/3d/tvj2vzsjoi
f(x,y,z) = y^2 - x^7*z
display( f.gradient() )
solve( list(f.gradient()), (x,y,z) )


# === Code Cell 8 ===

# Gathmann example 2.13
intersection_number( y^2-x^3, x^2-y^3 )


# === Code Cell 9 ===

A.<x,y> = AffineSpace(CC, 2)
X = A.subscheme([ y^2-x^3 ])
Y = A.subscheme([ x^2-y^3 ])
Q1 = Y([0,0])
Q1.intersection_multiplicity(X)


# === Code Cell 10 ===

Q1 = Y([1,1])
Q1.intersection_multiplicity(X)

