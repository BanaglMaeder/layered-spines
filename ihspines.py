import numpy as np
import copy

# Floor functions is used for certain perversity functions
from math import floor

from numpy.linalg import norm

# Delaunay triangulation used to generate Delaunay-Vietoris-Rips complexes
from scipy.spatial import Delaunay

###############################################################################
###########################Z/2Z SMITH NORMAL FORM##############################
###############################################################################


def AddRows(M, i, j):
   N = copy.deepcopy(M)
   N[i] = (M[i] + M[j]) % 2
   return N

def ExchangeRows(M, i, j):
    N = copy.deepcopy(M)
    N[[i,j]] = M[[j,i]] 
    return N

def ExchangeCols(M, i, j):
    N = copy.deepcopy(M)
    N[:, [i,j]] = N[:, [j,i]] 
    return N

def AddCols(M, i, j):
    N = copy.deepcopy(M)
    N[:, i] = (M[:, i] +  M[:, j] ) % 2
    return N  


def SNF(M, i=0):
    m, n = M.shape
        
    IndOnes = np.where(M[i:, i:] == 1)

    if IndOnes[0].size:
        j,k = IndOnes[0][0]+i, IndOnes[1][0]+i

        if (j,k) != (i,i):
            M = ExchangeRows(M, i, j)
            M = ExchangeCols(M, i, k)

        for l in range(i+1, m):
            if M[l,i] == 1:
                M = AddRows(M, l, i)
        
        for h in range(i+1, n):
            if M[i,h] == 1:
                M = AddCols(M, h, i)
            
        M = SNF(M, i+1)
    
    return M

###############################################################################
########################SIMPLICIAL OPERATIONS##################################
###############################################################################

#Computing the dimension of a simpl. cplx.
def ComplexDimension(C):
    
    if not C:
        return -100

    for k in range(len(C)-1,-1,-1):
        if len(C[k]):
            return k
    return -100

# SimplexIntersections returns the "largest" face of given simpl. s and t
def SimplexIntersection(s, t):
    return list(np.intersect1d(sorted(s),sorted(t)))

# function tests whether a simplex is contained in a simpl. cplx or not
def inArray(arr, list_of_arr):
     for elem in list_of_arr:
        if np.array_equal(sorted(arr), sorted(elem)):
            return True
     return False
 

# ComplexIntersection returns a list of simplices (not a complex but in a
# similar format) of the simplices in the simpl. cplx. K that are faces of the 
# simplex s. The special format of the output allows us to apply the function
# ComplexDimension to it.
def ComplexIntersection(s,K):
    
    k = len(s)-1
    n = ComplexDimension(K)
    
    if(k <= n):
        if inArray(sorted(s),K[k]):
            return sorted([s])
    
    inter = []
    
    for i in range(0,min(k+1,n+1)):
        inter = inter + [[]]
        for t in K[i]:
            u = np.intersect1d(sorted(s),sorted(t)).tolist()
            if (len(u) and not(inArray(sorted(u),inter[i])) and len(u)-1 == i):
                inter[i].append(u)
    return inter

###############################################################################
##########################INTERSECTION CHAIN COMPLEX###########################
###############################################################################

# Some common perversity functions
def TopP(k):
    if k < 2:
        return 0
    else:
        return k-2

def ZeroP(k):
    return 0
    
def LowMidP(k):
    if k < 2 :
        return 0
    else:
        return floor((k-2)/2)

def UpMidP(k):
    if k < 2:
        return 0
    else:
        return floor((k-1)/2)

def minus(k):
    return -1


# isProper decides whether or not a given simplex s is proper in the sense of
# corresponding to a simplicial chain that is transverse +- perversity
# Note: we allow a simplex to be proper even if it's boundary is not. The
# output is later on used to determine intersection chains.
def isProper(s, strata, p):
    
    if (p == "0"):
        p = ZeroP
    
    if (p == "t"):
        p = TopP
    
    if (p == "m"):
        p = LowMidP
        
    if (p == "n"):
        p = UpMidP
    
    if (p == "-1"):
        p = minus

    j = len(s)-1
    n = ComplexDimension(strata[0])
    
    for i in range(1,len(strata)):
        
        k = n - ComplexDimension(strata[i])
        
        dimIntersection = ComplexDimension(ComplexIntersection(s,strata[i]))
        
        if (dimIntersection > (j - k + p(k))):
            return False
        
    return True

# IS takes a simpl. cplx. C with specified stratification strata and a 
# perversity p. It returns the simplices in C that are proper (in the above
# sense) and the ones that are not proper in two separate lists.
def IS(C,strata,p):
    
    CP = []
    CIP = []
        
    for i in range(len(C)):
        CP.append([])
        CIP.append([])
        for x in C[i]:
            if isProper(x,strata,p):
                CP[i].append(x)
            else:
                CIP[i].append(x)
    
    return CP,CIP

# In the following we define some functions to perform a matrix reduction
# algorithm. This will be used to identify all simplicial intersection
# chains, also the non elementary ones.

def low(M,j):
    col = np.nonzero(M[:,j])
    if len(col[0]) == 0:
        return -1
                 
    return np.where(M[:,j] == M[:,j][col[0]].min())[0].max()

# The input for the function MatrixReduction is a matrix M and an integer k.
# This routine executes elementary column transformations from left to right    
# in order to eliminate nonzero entries below the row index k.
# The output includes the matrix M in reduced form and a lists columns whose
# entries below index k are all zero. This process works in Z/2Z only.
def MatrixReduction(M,k):
    
    comb = []
    for t in range(0,M.shape[1]):
        comb.append([t])
    ProperComb = []
    
    stop = False
    while not stop:
        count = 0
        for j in range(M.shape[1]-1,-1,-1):
            if low(M,j) > k:
                for i in range(j-1,-1,-1):
                    if low(M,i) == low(M,j) and low(M,j) > k:
                        M[:,j] = M[:,j]+M[:,i]
                        comb[j]= comb[j] + comb[i]
                        count = count+1
                        M = M%2
                  
        if count == 0:
            stop = True
                
    for j in range(0,M.shape[1]):
        if low(M,j) <= k:
            ProperComb.append(comb[j])
    
    return M, ProperComb

# The function IC accepts a simpl. cplx. C, a stratification strata 
# and a perversity p. The output includes the perversity p Intersection
# Chain Complex associated with the initial complex C. The filtration is
# specified by strata. Furthermore, IC also returns the Betti numbers of
# perversity p intersection homology.
def IC(C,strata,p):
    
    CP, CIP = IS(C,strata,p)
    
    n = len(CP)-1
    
    ranks = [0]
    
    # list for the resulting Intersection Chain Complex
    ICC = []
    for i in range(0,len(CP)):
        ICC.append([])
        
    for v in CP[0]:
        ICC[0].append([v])
    
    for i in range(n,0,-1): 
        ns1 = len(CP[i])
        # Note: If there are no improper simplices in this dimension there is 
        # nothing to do
        numImprop = len(CIP[i-1])
        
        aC = CP[i-1] + CIP[i-1]
        
        # Setting up the binary incidence matrix following the order in aC.
        M = np.zeros((len(C[i-1]), ns1), dtype=int)
        for j in range (0, ns1):
            s = CP[i][j]
            facets = []
            for k in range (0, i+1):
                f = s.copy()
                del f[k]
                facets = facets + [f]
    
            for k in range (0, len(C[i-1])):
                if aC[k] in facets:
                    M[k,j] = 1
        redM = MatrixReduction(M,len(C[i-1])-1-numImprop)
        # We determine the intersection chain complex with redM[1].
        # The list redM[1] contains indices corresponding to the proper 
        # i-simplices that make as sum an allowable simpl. chain
        for l in redM[1]:
            c = []
            for ind in l:
                c.append(CP[i][ind])
            ICC[i].append(c)
            
        # Next, we calculate the Betti numbers via the rank of a reduced matrix
        B = redM[0]
        A = np.zeros((len(C[i-1]), ns1), dtype=int)
        for j in range(0,B.shape[1]):
            if low(B,j) <= len(C[i-1])-1-numImprop:
                A[:,j] = B[:,j]
        shapeA = np.shape(A)
        if shapeA[0] == 0 or shapeA[1] == 0:
            R = 0
        else:
            A_snf = SNF(A) 
            R = 0
            for i in range(0,min(shapeA[0],shapeA[1])):
                if A_snf[i,i] == 1:
                    R = R+1
        ranks.append(R)
        
    ranks.append(0)
    
    ranks = ranks[::-1]
    
    BettiNumbers = []
    
    n = len(ICC)
    
    for i in range(n):
        Betti = len(ICC[i])-ranks[i]-ranks[i+1]
        BettiNumbers.append(Betti)
        
        
    return ICC, BettiNumbers



# Auxiliary function to check whether or not a given simpl. cplx. represents a
# pseudomanifold
def isPseudomanifold(C):

    n = ComplexDimension(C)
    countlist = []
    for i in range(0,n):
        for s in C[i]:
            count = 0
            for t in C[n]:
                if len(SimplexIntersection(s,t))==len(s):
                    count = count +1
            if count == 0:
                return False
            if i == n-1:
                countlist.append(count)
                if count != 2:
                    return [False,countlist]
    
    return True


###############################################################################
###################SIMPLICIAL COMPLEXES FROM POINT CLOUDS######################
###############################################################################

# DelVR complex, compare do DelCech complex as in Bauer & Edelsbrunner 2017.
# Currently restricted to point clouds xyz of dimension <= 3.
# We employed the function Delaunay from the scipy.spatial package to realize
# Delaunay triangulations.
def DelaunayComplex(xyz,r):
    
    dim=len(xyz[0])
    
    edges = []
    triangles = []
    tetrahedra = []
    
    lengthpc = len(xyz)
    
    vertices = [[i] for i in range(0,lengthpc)]
    
    pcnp = np.array(xyz)
    
    delaunay = Delaunay(xyz).simplices
    
    # First we construct Delaunay triangulation and then select  simplices
    # whose vertices lie pairwise closer than distance r to each other.
    if dim==2:
        DelE = []
        DelTr = delaunay
        for i in range(0, len(DelTr)):
            triple = DelTr[i]
            triple.sort()
            DelE.append(list([triple[0], triple[1]]))
            DelE.append(list([triple[0], triple[2]]))
            DelE.append(list([triple[1], triple[2]]))
            
        # DelE may contain duplicate simplices. So we need to remove these
        # duplicates:
        auxtup = [tuple(s) for s in DelE]
        auxset = set(auxtup)
        auxlist = list(auxset)
        DelE = [list(t) for t in auxlist]
            
    if dim==3:
        DelE = []
        DelTr = []
        DelTe = delaunay
        for i in range(0, len(DelTe)):
            quad = DelTe[i]
            quad.sort()
            DelTr.append(list([quad[0], quad[1], quad[2]]))
            DelTr.append(list([quad[0], quad[1], quad[3]]))
            DelTr.append(list([quad[0], quad[2], quad[3]]))
            DelTr.append(list([quad[1], quad[2], quad[3]]))
        auxtup = [tuple(s) for s in DelTr]
        auxset = set(auxtup)
        auxlist = list(auxset)
        DelTr = [list(t) for t in auxlist]

        for i in range(0, len(DelTr)):
            triple = DelTr[i]
            DelE.append(list([triple[0], triple[1]]))
            DelE.append(list([triple[0], triple[2]]))
            DelE.append(list([triple[1], triple[2]]))
        auxtup = [tuple(s) for s in DelE]
        auxset = set(auxtup)
        auxlist = list(auxset)
        DelE = [list(t) for t in auxlist]
        

    for e in DelE:
        i = e[0]
        j = e[1]
        distance = norm(pcnp[i] - pcnp[j])
        if(r >= distance/2):
                edges.append([i, j])        
                    
    for tri in DelTr:
        i = tri[0]
        j = tri[1]
        k = tri[2]
        M = max(norm(pcnp[j]-pcnp[k]), 
                norm(pcnp[i]-pcnp[j]),
                norm(pcnp[i]-pcnp[k]))
        if(r >= M/2):
            triangles.append([i, j, k])        
                    
    if dim == 3:
        for tet in DelTe:
            i = tet[0]
            j = tet[1]
            k = tet[2]
            l = tet[3]
            M = max(norm(pcnp[i]-pcnp[j]),
                    norm(pcnp[i]-pcnp[k]),
                    norm(pcnp[i]-pcnp[l]),
                    norm(pcnp[j]-pcnp[k]),
                    norm(pcnp[j]-pcnp[l]),
                    norm(pcnp[k]-pcnp[l]))
            if(r >= M/2):
                tetrahedra.append([i, j, k,l])     

    return [vertices,edges,triangles,tetrahedra]


# The function VRComplex calculates the Vietoris-Rips complex of a 
# point cloud xyz for the radius r. Currently the complex is restricted
# to dimension 3.

def VRComplex(xyz,r):
    
    lengthpc = len(xyz)
    
    
    pcnp = [np.array(x) for x in xyz]  
    
    
    VR0S = [[i] for i in range (0, lengthpc)]

   
    Diameter = 2*r
    
    VR1S = []

    for i in range(0, lengthpc):
        for j in range (i+1, lengthpc):
            if norm(pcnp[i] - pcnp[j]) < Diameter:
                VR1S = VR1S + [[i,j]]


    VR2S = []
    
    for s1 in VR1S:
        for i in range (0, lengthpc):
            j = s1[0]
            k = s1[1]
            if i != j and i != k:
                x = pcnp[j]
                y = pcnp[k]
                nx = norm(pcnp[i] - x)
                ny = norm(pcnp[i] - y)
                if nx < Diameter and ny < Diameter:
                    # Build a 2-simplex s2 with vertices i,j,k:
                    s2 = [i,j,k]
                    # s2 need not be an >oriented< 2-simplex; we first
                    # need to sort the vertices in ascending order:
                    s2.sort()
                    # add the oriented 2-simplex s2 to the
                    # Vietoris-Rips complex:
                    VR2S = VR2S + [s2]        
                        

    # VR2S  may contain duplicate simplices. So we need to remove these
    # duplicates:
    auxtup = [tuple(s) for s in VR2S]
    auxset = set(auxtup)
    auxlist = list(auxset)
    VR2S = [list(t) for t in auxlist]



    VR3S = []
    
    # We compute the 3-simplices of the Vietoris-Rips complex.
    # This operation is quadratic in the number of data points/2-simplices.
    # s2 ranges over all 2-simplices:
    for s2 in VR2S:
        for i in range (0, lengthpc):
            j = s2[0]
            k = s2[1]
            l = s2[2]
            if i != j and i != k and i != l:
                x = pcnp[j]
                y = pcnp[k]
                z = pcnp[l]
                nx = norm(pcnp[i] - x)
                ny = norm(pcnp[i] - y)
                nz = norm(pcnp[i] - z)
                if nx < Diameter and ny < Diameter and nz < Diameter:    
                    # Build a 3-simplex s3 with vertices i,j,k,l:
                    s3 = [i,j,k,l]
                    # s3 need not be an >oriented< 3-simplex; we first
                    # need to sort the vertices in ascending order:
                    s3.sort()
                    # add the oriented 3-simplex s3 to the
                    # Vietoris-Rips complex:
                    VR3S = VR3S + [s3]

    auxtup = [tuple(s) for s in VR3S]
    auxset = set(auxtup)
    auxlist = list(auxset)
    VR3S = [list(t) for t in auxlist]


    
    return [VR0S,VR1S,VR2S,VR3S]     
         
###############################################################################
###############################(LAYERED) SPINES################################
###############################################################################

# Auxillary function to check if a given simplex t is principal in a simpl.
# cplx. C
def isPrincipal(C,t):
    
    k = len(t)-1
    
    if k == ComplexDimension(C):
        return True
    
    for s in C[k+1]:
        if len(SimplexIntersection(s,t)) == t:
            return False

    return True

# Princ will take a simpl. cplx. C and a simplex s of C.
# The output is the set of principal cofaces of s in C
def Princ(C,s):
    
    n = ComplexDimension(C)
    k = len(s)-1
    if(k == n):
        return []
    
    count = 0
    p = []
    
    for t in C[k+1]:
        if len(SimplexIntersection(s,t))-1  == k:    
            count = count+1
            if count > 1:
                return []
            if isPrincipal(C,t):
                p.append(t)

    return p

    

# isAdmissable is an auxiliary function to check the extra condition for an 
# intermediate collapse to be elementary
def isAdmissible(s,p,S):

    T = ComplexIntersection(p,S)
    
    for t in T:
        for r in t:
            inter = len(SimplexIntersection(r,s))
            if (inter == 0 or inter == len(s)):
                return False
    
    return True 


# Function to realise a Collapse. Only applied if condition for an elementary 
# collapse are fulfilled.
def ElemCollapse(C,s,p):
    
    k = len(p)-1
    
    C[k].remove(p)
    C[k-1].remove(s)
    
    return C
# The Function Spine computes the layered spine of a given Simpl. Cplx.(Cplx) 
# with resp. to S0 and C0. If one of them is empty (this has to be specified)
# the result will be a spine of Cplx in the usual sense.

def Spine(Cplx, S0 , C0):
    
    # We create deep copies to not change the input
    # Note: This doubles the required memory
    K = copy.deepcopy(Cplx)
    n = ComplexDimension(K)
    S = copy.deepcopy(S0)
    C = copy.deepcopy(C0)
    IM = [[]]

        
    for i in range(1,n+1):
        
    # Every increment will add a list to S, C and IM to be the i-th Skeleton
        S = S + [[]]
        C = C + [[]]
        IM = IM + [[]]
        for t in K[i]:
            
        # Here we check if all vertices of a simplex t lie in S, C, or partly
        # in both (i.e. in IM)
            if ComplexDimension(S) >= 0:
                a = len(ComplexIntersection(t,S0)[0])
            else:
                a = -100
            
            if ComplexDimension(C) >= 0:
                b = len(ComplexIntersection(t,C0)[0])
            else:
                b = -100
                    
            if  a == len(t):
                S[i].append(t)
                
            if b == len(t):
                C[i].append(t)
                
            if a != len(t) and b!=len(t):
                IM[i].append(t)
               
    #S-Collapse
    stop = False
    # After the execution of an ElemCollapse we have to go through the
    # remaining simplices because simplices can become free after a collapse.
    while not stop:
        count = 0
        for i in range(min(ComplexDimension(K)-1,ComplexDimension(S)-1),-1,-1):
            # Creating a copy of S to iterate over
            Scopy = copy.deepcopy(S)
            for s in Scopy[i]:
            # We search the i-th skeleton for free faces                
                princ_s = Princ(K,s)                
                # s is free if there is exactly one princ coface and none other
                # princ_s either contains the unique principal coface of s
                # if s is free or it is an empty list
                if len(princ_s) == 1:
                    if princ_s[0] in S[i+1]:

                        K = ElemCollapse(K,s,princ_s[0])
                        S = ElemCollapse(S,s,princ_s[0])
                        count = count +1
                   
        # If no collapse has been executed we are done and break the while-loop 
        if count == 0:
            stop = True

    #C-Collapses
    stop = False
    while not stop:
        count = 0
        for i in range(min(ComplexDimension(K)-1,ComplexDimension(C)-1),-1,-1):
            Ccopy = copy.deepcopy(C)
            for c in Ccopy[i]:
                princ_c = Princ(K,c)
                if len(princ_c) == 1:
                    if princ_c[0] in C[i+1]:

                        K = ElemCollapse(K,c,princ_c[0])
                        C = ElemCollapse(C,c,princ_c[0])
                        count = count +1
                    
        if count == 0:
            stop = True
                    
    #Intermediate-Collapses
    stop = False
    while not stop:
        count = 0
        for j in range(min(ComplexDimension(K)-1,ComplexDimension(IM)-1),-1,-1):
            IMcopy = copy.deepcopy(IM)
            for i in IMcopy[j]:
                princ_i = Princ(K,i)
                if len(princ_i) == 1:
                    # Note: we have to check an extra condition for
                    # intermediate collapses to be elementary
                    if isAdmissible(i, princ_i[0], S):
                        K = ElemCollapse(K,i,princ_i[0])
                        IM = ElemCollapse(IM,i,princ_i[0])
                        count = count + 1
                    
        if count == 0:
            stop = True
    
    
    #C-Collapses
    stop = False
    while not stop:
        count = 0
        for i in range(min(ComplexDimension(K)-1,ComplexDimension(C)-1),-1,-1):
            Ccopy = copy.deepcopy(C)
            for c in Ccopy[i]:
                princ_c = Princ(K,c)
                if len(princ_c) == 1:
                    if princ_c[0] in C[i+1]:
                        K = ElemCollapse(K,c,princ_c[0])
                        C = ElemCollapse(C,c,princ_c[0])
                        count = count +1   
                    
        if count == 0:
            stop = True
            
    return K
