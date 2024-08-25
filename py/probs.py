print("test")

P = [0.3, 0.7]
Q = [0.5, 0.5] 

print(P, Q)

from scipy.special import rel_entr

print(sum(rel_entr(P, Q)))
