# layered-spines
Short description of the program

## Examples

```
from ihspines import IC

# Triangulation for Pinched Torus
PT2 = [[[1],[2],[3],[4],[5],[6],[7]],[[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],
        [2,4],[2,6],[2,7],[3,4],[3,5],[3,6],[3,7],[4,6],[4,7],[5,6],[5,7],
        [6,7]],[[1,2,4],[1,3,4],[3,4,6],[3,5,6],[1,3,7],[1,5,7]
        ,[3,5,7],[2,4,7],[2,6,7],[4,6,7],[1,2,6],[1,5,6]]]

# Vertex [7] represents the singular point.
#A filtration for PT2 is given by the list 
filt =[PT2,[[[7]]]]

# For the zero perversity we compute the Intersection chain complex and 
# the Betti numbers of intersection homology for PT2:
ICPT2, IHBettiPT2 = IC(PT2,filt,"0")
>>> ICPT2
    [[[[1]], [[2]], [[3]], [[4]], [[5]], [[6]]],
    [[[1, 2]],
     [[1, 3]],
     [[1, 4]],
     [[1, 5]],
     [[1, 6]],
     [[2, 4]],
     [[2, 6]],
     [[3, 4]],
     [[3, 5]],
     [[3, 6]],
     [[4, 6]],
     [[5, 6]]],
    [[[1, 2, 4]],
     [[1, 3, 4]],
     [[3, 4, 6]],
     [[3, 5, 6]],
     [[3, 5, 7], [1, 5, 7], [1, 3, 7]],
     [[4, 6, 7], [2, 6, 7], [2, 4, 7]],
     [[1, 2, 6]],
     [[1, 5, 6]]]]

>>> ICBettiPT2
    [1, 0, 1]
```

```
from ihspines import Spine

# Simplicial complex generated by a 3-simplex
F1 = [[[1],[2],[3],[4]],[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]],[[1,2,3],[1,2,4],
       [1,3,4],[2,3,4]],[[1,2,3,4]]]
    
# Example computation for a layered spine with 
S0 = [[[1],[2]]]
C0 = [[[3],[4]]]

LaySpineF1 = Spine(F1,S0,C0)
>>> LaySpineF1
    [[[1], [2], [4]], [[1, 2], [1, 4], [2, 4]], [[1, 2, 4]], []]
    
# Example computation for a spine with 
S0 = [[]]
C0 = [[[1],[2],[3],[4]]]

SpineF1 = Spine(F1,S0,C0)
>>> SpineF1
    [[[3]], [], [], []]

```
