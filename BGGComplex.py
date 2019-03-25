from itertools import groupby,chain

#I don't know what sage module to import to make matrix(...) work (I want to coerce the list of simple roots into a matrix)
#from sage.combinat.root_system.weyl_group import  WeylGroup
#from sage.graphs.digraph import DiGraph
#from sage.matrix import *

from sage.all import *

class BGGComplex:
    """A class encoding all the things we need of the BGG complex"""
    def __init__(self, root_system):
        self.W = WeylGroup(root_system)
        self.S = self.W.simple_reflections()
        self.T = self.W.reflections()
        
        self._compute_weyl_dictionary()
        self._construct_BGG_graph()
        
        self.rho = self.W.domain().rho()
        self.simple_roots = self.W.domain().simple_roots().values()
        self.zero_root = self.W.domain().zero()
        
    def _compute_weyl_dictionary(self):
        """Construct a dictionary enumerating all of the elements of the Weyl group. The keys are strings encoding a word in the simple reflections, and the values are elements of the Weyl group"""
        self.WeylDic={"":self.W.one()} #start with the identity element
        word_length=0
        while len(self.WeylDic)<self.W.order(): #continue until we enumerated all the elements of the Weyl group
            #we inductively find elements of longer word lengths by multipliying them by all the simple relflections. 
            #We then keep only the elements we haven't found yet.
            #This encodes every element of they Weyl group as a word in the simple reflections (but this is not unique)
            BGGColumn=[s for s in self.WeylDic.keys() if len(s)==word_length] 
            for w in BGGColumn:
                for s in self.W.index_set():
                    test_element= self.S[s]*self.WeylDic[w]
                    if test_element not in self.WeylDic.values():
                        self.WeylDic[str(s)+w]=test_element
            word_length+=1
        self.WeylDicReverse=dict([[v,k] for k,v in self.WeylDic.items()])
    
    def _construct_BGG_graph(self):
        "Find all the arrows in the BGG Graph. There is an arrow w->w' if len(w')=len(w)+1 and w' = t.w for some t in T."
        self.arrows=[]
        for w in self.WeylDic.keys():
            for t in self.T:
                product_word = self.WeylDicReverse[t*self.WeylDic[w]]
                if len(product_word)==len(w)+1:
                    self.arrows+=[(w,product_word)]
        self.graph= DiGraph(self.arrows)
    
    def plot_graph(self):
        """Create a pretty plot of the BGG graph. Each word length is encoded by a different color. Usage: _.plot_graph().plot()"""
        BGGVertices=sorted(self.WeylDic.keys(),key=len) 
        BGGPartition=[list(v) for k,v in groupby(BGGVertices,len)]

        BGGGraphPlot = self.graph.to_undirected().graphplot(partition=BGGPartition,vertex_labels=None,vertex_size=30)
        return BGGGraphPlot

    def find_cycles(self):
        """Find all the admitted cycles in the BGG graph. An admitted cycle consists of two paths a->b->c and a->b'->c, where the word length increases by 1 each step. The cycles are returned as tuples (a,b,c,b',a)."""

        #for faster searching, make a dictionary of pairs (v,[u_1,...,u_k]) where v is a vertex and u_i are vertices such that
        #there is an arrow v->u_i
        outgoing={k:map(lambda x:x[1],v) for k,v in groupby(self.arrows,lambda x: x[0])}
        outgoing[max(self.WeylDic.keys(),key=lambda x: len(x))]=[]
        
        # make a dictionary of pairs (v,[u_1,...,u_k]) where v is a vertex and u_i are vertices such that
        #there is an arrow u_i->v
        incoming={k:map(lambda x:x[0],v) for k,v in groupby(self.arrows,lambda x: x[1])}
        incoming['']=[]

        #enumerate all paths of length 2, a->b->c, where length goes +1,+1
        self.cycles=chain.from_iterable([[a+(v,) for v in outgoing[a[-1]]] for a in self.arrows])
        
        #enumerate all paths of length 3, a->b->c->b' such that b' != b and length goes +1,+1,-1
        self.cycles=chain.from_iterable([[a+(v,) for v in incoming[a[-1]] if v != a[1]] for a in self.cycles])
        
        #enumerate all cycles of length 4, a->b->c->b'->a such that b'!=b and length goes +1,+1,-1,-1
        self.cycles=[a+(a[0],) for a in self.cycles if a[0] in incoming[a[-1]]]

        return self.cycles

    def _weight_to_tuple(self,weight):
        """Decompose a weight into a tuple encoding the weight as a linear combination of the simple roots"""
        b=weight.to_vector()
        b=matrix(b).transpose()
        A=[list(a.to_vector()) for a in self.simple_roots]
        A=matrix(A).transpose()
        return transpose(A.solve_right(b)).list()

    def _tuple_to_weigth(self,t):
        """Turn a tuple encoding a linear combination of simple roots back into a weight"""
        return sum(a*b for a,b in zip(t,self.simple_roots))

    def dot_action(self,reflection,weight_tuple):
        """Compute the dot action of a reflection on a weight. The reflection should be an element of the Weyl group self.W and the weight should be given as a tuple encoding it as a linear combination of simple roots."""
        weight = self._tuple_to_weigth(weight_tuple)
        new_weight= reflection.action(weight+self.rho)-self.rho
        return self._weight_to_tuple(new_weight)
    
    def longest_element(self):
        """Returns the longest element of the Weyl groups"""
        return max(self.WeylDic.keys(),key=len)
        