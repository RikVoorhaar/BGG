"""
This module implements the BGG complex.

Example usage:

.. code:: python

    >>> bgg = BGGComplex('A2')
    >>> bgg.plot_graph() # Show the Bruhat graph
    >>> bgg.display_maps((0,0)) # Give the maps of the BGG complex (without sing)
    >>> bgg.compute_signs() # Give signs in the complex, turning maps into differential

"""

from itertools import groupby, chain

import os
import pickle

try:
    import sage.all  # pylint: disable=unused-import
except ImportError:
    import sage.all__sagemath_modules

from sage.rings.rational_field import QQ
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.algebras.lie_algebras.lie_algebra import LieAlgebra
from sage.combinat.root_system.weyl_group import WeylGroup
from sage.graphs.digraph import DiGraph
from sage.modules.free_module_element import vector

from numpy import array

from .compute_signs import compute_signs
from .compute_maps import BGGMapSolver
from .pbw import PoincareBirkhoffWittBasis
from .weight_set import WeightSet

from collections import defaultdict

from IPython.display import display, Math


class BGGComplex:
    """A class encoding all the things we need of the BGG complex.
    
    Parameters
    ----------
    root_system : str
        String encoding the Dynkin diagram of the root system (e.g. `'A3'`)
    pickle_directory : str (optional)
            Directory where to store `.pkl` files to save computations
            regarding maps in the BGG complex. If `None`, maps are not saved. (default: `None`)


    Attributes
    -------
    root_system : str
        String encoding Dynkin diagram.
    W : WeylGroup
        Object encoding the Weyl group.
    LA : LieAlgebraChevalleyBasis
        Object encoding the Lie algebra in the Chevalley basis over Q
    PBW : PoincareBirkhoffWittBasis
        Poincaré-Birkhoff-Witt basis of the universal enveloping 
        algebra. The maps in the BGG complex are elements of this basis. This package uses
        a slight modification of the PBW class as implemented in Sagemath proper.
    lattice : RootSpace
        The space of roots
    S : FiniteFamily
        Set of simple reflections in the Weyl group
    T : FIniteFamily
        Set of reflections in the Weyl group
    cycles : List 
        List of 4-tuples encoding the length-4 cycles in the Bruhat graph
    simple_roots : List
        (Ordered) list of the simple roots as elements of `lattice`
    rank : int
        Rank of the root system
    neg_roots : list
        List of the negative roots in the root system, encoded as `numpy.ndarray`.
    alpha_to_index : dict
        Dictionary mapping negative roots (as elements of `lattice`) to
        an ordered index encoding the negative root as basis element of the lie algebra $n$
        spanned by the negative roots.
    zero_root : element of `lattice`
        The zero root
    rho : element of `lattice`
        Half the sum of all positive roots
    """

    def __init__(self, root_system, pickle_directory=None):
        self.root_system = root_system
        self.W = WeylGroup(root_system)
        self.domain = self.W.domain()
        self.LA = LieAlgebra(QQ, cartan_type=root_system)
        self.PBW = PoincareBirkhoffWittBasis(self.LA, None, "PBW", cache_degree=5)
        # self.PBW = self.LA.pbw_basis()
        self.PBW_alg_gens = self.PBW.algebra_generators()
        self.lattice = self.domain.root_system.root_lattice()
        self.S = self.W.simple_reflections()
        self.T = self.W.reflections()
        self.signs = None
        self.cycles = None

        self._compute_weyl_dictionary()
        self._construct_BGG_graph()

        self.find_cycles()

        self.simple_roots = self.domain.simple_roots().values()
        self.rank = len(self.simple_roots)

        # for PBW computations we need to put the right order on the negative roots.
        # This order coincides with that of the sagemath source code.
        lie_alg_order = {k: i for i, k in enumerate(self.LA.basis().keys())}
        ordered_roots = sorted(
            [self._weight_to_alpha_sum(r) for r in self.domain.negative_roots()],
            key=lambda rr: lie_alg_order[rr],
        )
        self.neg_roots = [-self._alpha_sum_to_array(r) for r in ordered_roots]

        self.alpha_to_index = {
            self._weight_to_alpha_sum(-self._tuple_to_weight(r)): i
            for i, r in enumerate(self.neg_roots)
        }
        self.zero_root = self.domain.zero()

        self.pickle_directory = pickle_directory
        if pickle_directory is None:
            self.pickle_maps = False
        else:
            self.pickle_maps = True

        if self.pickle_maps:
            self._maps = self._read_maps()
        else:
            self._maps = dict()

        self.rho = self.domain.rho()

        self._action_dic = dict()
        for s, w in self.reduced_word_dic.items():
            self._action_dic[s] = {
                i: self._weight_to_alpha_sum(w.action(mu))
                for i, mu in dict(self.domain.simple_roots()).items()
            }
        self._rho_action_dic = dict()
        for s, w in self.reduced_word_dic.items():
            self._rho_action_dic[s] = self._weight_to_alpha_sum(
                w.action(self.rho) - self.rho
            )

    def _compute_weyl_dictionary(self):
        """Construct a dictionary enumerating all of the elements of the Weyl group."""
        self.reduced_word_dic = {
            "".join([str(s) for s in g.reduced_word()]): g for g in self.W
        }
        self.reduced_word_dic_reversed = dict(
            [[v, k] for k, v in self.reduced_word_dic.items()]
        )
        self.reduced_words = sorted(
            self.reduced_word_dic.keys(), key=len
        )  # sort the reduced words by their length
        long_element = self.W.long_element()
        self.dual_words = {
            s: self.reduced_word_dic_reversed[long_element * w]
            for s, w in self.reduced_word_dic.items()
        }  # the dual word is the word times the longest element

        self.column = defaultdict(list)
        max_len = 0
        for red_word in self.reduced_words:
            length = len(red_word)
            max_len = max(max_len, length)
            self.column[length] += [red_word]
        self.max_word_length = max_len

    def _construct_BGG_graph(self):
        """Find all the arrows in the BGG Graph.

        There is an arrow w->w' if len(w')=len(w)+1 and w' = t.w for some t in T.
        """
        self.arrows = []
        for w in self.reduced_words:
            for t in self.T:
                product_word = self.reduced_word_dic_reversed[
                    t * self.reduced_word_dic[w]
                ]
                if len(product_word) == len(w) + 1:
                    self.arrows += [(w, product_word)]
        self.arrows = sorted(
            self.arrows, key=lambda t: len(t[0])
        )  # sort the arrows by the word length
        self.graph = DiGraph(self.arrows)

    def plot_graph(self):
        """Create a pretty plot of the BGG graph, with vertices colored by word lenght."""
        BGGVertices = sorted(self.reduced_words, key=len)
        BGGPartition = [list(v) for k, v in groupby(BGGVertices, len)]

        BGGGraphPlot = self.graph.to_undirected().graphplot(
            partition=BGGPartition, vertex_labels=None, vertex_size=30
        )
        display(BGGGraphPlot.plot())

    def find_cycles(self):
        """Find all the admitted cycles in the BGG graph.
        
        An admitted cycle consists of two paths a->b->c and a->b'->c, 
        where the word length increases by 1 each step. 
        The cycles are returned as tuples (a,b,c,b',a).
        """
        # only compute cycles if we haven't yet done so already
        if self.cycles is None:
            # for faster searching, make a dictionary of pairs (v,[u_1,...,u_k]) where v is a vertex and u_i
            # are vertices such that there is an arrow v->u_i
            first = lambda x: x[0]
            second = lambda x: x[1]
            outgoing = {
                k: list(map(second, v))
                for k, v in groupby(sorted(self.arrows, key=first), first)
            }
            # outgoing[max(self.reduced_words,key=lambda x: len(x))]=[]
            outgoing[self.reduced_word_dic_reversed[self.W.long_element()]] = []

            # make a dictionary of pairs (v,[u_1,...,u_k]) where v is a vertex and u_i are vertices such that
            # there is an arrow u_i->v
            incoming = {
                k: list(map(first, v))
                for k, v in groupby(sorted(self.arrows, key=second), second)
            }
            incoming[""] = []

            # enumerate all paths of length 2, a->b->c, where length goes +1,+1
            self.cycles = chain.from_iterable(
                [[a + (v,) for v in outgoing[a[-1]]] for a in self.arrows]
            )

            # enumerate all paths of length 3, a->b->c->b' such that b' != b,
            # b<b' in lexicographic order (to avoid duplicates) and length goes +1,+1,-1
            self.cycles = chain.from_iterable(
                [[a + (v,) for v in incoming[a[-1]] if v > a[1]] for a in self.cycles]
            )

            # enumerate all cycles of length 4, a->b->c->b'->a such that b'!=b and length goes +1,+1,-1,-1
            self.cycles = [a + (a[0],) for a in self.cycles if a[0] in incoming[a[-1]]]

        return self.cycles

    def compute_signs(self, force_recompute=False):
        """Compute signs making making product of signs around all squares equal to -1.

        Returns
        -------
        dict[tuple(str,str), int]
            Dictionary mapping edges in the Bruhat graph to {+1,-1}
        """
        if not force_recompute:
            if self.signs is not None:
                return self.signs

        self.signs = compute_signs(self)
        return self.signs

    def compute_maps(self, root, column=None, check=False, pbar=None):
        """Compute the (unsigned) maps of the BGG complex for a given weight.

        Parameters
        ----------
        root : tuple(int)
            root for which to compute the maps.
        column : int or `None` (default: `None`)
            Try to only compute the maps up to this particular column if not `None`. This
            is faster in particular for small or large values of `column`. 
        check : bool (default: False)
            After computing all the maps, perform a check whether they are correct for 
            debugging purposes.
        pbar : tqdm (default: None)
            tqdm progress bar to give status updates about the progress. If `None` this 
            feature is disabled.

        Returns
        -------
        dict mapping edges (in form ('w1', 'w2')) to elements of `self.PBW`.
        """
        # Convert to tuple to make sure root is hasheable
        root = tuple(root)

        # If the maps are not in the cache, compute them and cache the result
        if root in self._maps:
            cached_result = self._maps[root]
        else:
            cached_result = None
        MapSolver = BGGMapSolver(self, root, pbar=pbar, cached_results=cached_result)
        self._maps[root] = MapSolver.solve(column=column)
        if check:
            maps_OK = MapSolver.check_maps()
            if not maps_OK:
                raise ValueError(
                    "For root %s the map solver produced something wrong" % root
                )

        if self.pickle_maps:
            self._store_maps()

        return self._maps[root]

    def _read_maps(self):
        target_path = os.path.join(
            self.pickle_directory, self.root_system + r"_maps.pkl"
        )
        try:
            with open(target_path, "rb") as file:
                maps = pickle.load(file)
                return maps
        except IOError:
            return dict()

    def _store_maps(self):
        target_path = os.path.join(
            self.pickle_directory, self.root_system + r"_maps.pkl"
        )
        try:
            with open(target_path, "wb") as file:
                pickle.dump(self._maps, file, pickle.HIGHEST_PROTOCOL)
        except IOError:
            pass

    def _weight_to_tuple(self, weight):
        """Convert rootspace element to tuple."""
        b = weight.to_vector()
        b = matrix(b).transpose()
        A = [list(a.to_vector()) for a in self.simple_roots]
        A = matrix(A).transpose()

        return tuple(A.solve_right(b).transpose().list())

    def _weight_to_alpha_sum(self, weight):
        """Express a weight in the lattice as a linear combination of alpha[i]'s.
        
        These objects form the keys for elements of the Lie algebra, 
        and for factors in the universal enveloping algebra.
        """
        if type(weight) is not tuple and type(weight) is not list:
            tup = self._weight_to_tuple(weight)
        else:
            tup = tuple(weight)
        alpha = self.lattice.alpha()
        zero = self.lattice.zero()
        return sum((int(c) * alpha[i + 1] for i, c in enumerate(tup)), zero)

    def _alpha_sum_to_array(self, weight):
        output = array(vector(ZZ, self.rank))
        for i, c in weight.monomial_coefficients().items():
            output[i - 1] = c
        return output

    def _tuple_to_weight(self, t):
        """Turn a tuple encoding a linear combination of simple roots back into a weight."""
        return sum(int(a) * b for a, b in zip(t, self.simple_roots))

    def _is_dot_regular(self, mu):
        """Check if a weight is dot-regular by checking that it has trivial stabilizer under dot action.
        
        Parameters
        ----------
        mu : element of `self.lattice`
            Weight encoded as linear combination of `alpha[i]`, use `self._weight_to_alpha_sum()`
            to convert to this format
        """
        for w in self.reduced_words[1:]:
            if (
                self._dot_action(w, mu) == mu
            ):  # Stabilizer is non-empty, mu is not dot regular
                return False

        # No nontrivial stabilizer found
        return True

    def _make_dominant(self, mu):
        """Given a dot-regular weight, find the associated dominant weight.

        Parameters
        ----------
        mu : element of `self.lattice`
            Weight encoded as linear combination of `alpha[i]`, use `self._weight_to_alpha_sum()`
            to convert to this format

        Returns
        -------
        dominant weight encoded in same format as input
        """
        for w in self.reduced_words:
            new_mu = self._dot_action(w, mu)
            if new_mu.is_dominant():
                return new_mu, w

        # Nothing found
        raise Exception(
            "The weight %s can not be made dominant. Probably it is not dot-regular."
            % mu
        )

    # def compute_weights(self, weight_module):
    #     all_weights = weight_module.weight_dic.keys()

    #     regular_weights = []
    #     for mu in all_weights:
    #         if self._is_dot_regular(mu):
    #             mu_prime, w = self._make_dominant(mu)
    #             # mu_prime = self._weight_to_alpha_sum(mu_prime)
    #             # w = self.reduced_word_dic_reversed[w]
    #             regular_weights.append((mu, mu_prime, len(w)))
    #     return all_weights, regular_weights

    def _dot_action(self, w, mu):
        """Dot action of Weyl group on weights.

        Parameters
        ----------
        w : element of `self.W`
        mu : RootSpace.element_class
            Weight encoded as linear combination of `alpha[i]`, use `self._weight_to_alpha_sum()`
            to convert to this format

        """
        action = self._action_dic[w]
        mu_action = sum(
            [action[i] * int(c) for i, c in mu.monomial_coefficients().items()],
            self.lattice.zero(),
        )
        return mu_action + self._rho_action_dic[w]

    def display_pbw(self, f, notebook=True):
        """Typesets an element of PBW of the universal enveloping algebra with LaTeX.

        Parameters
        ----------
        f : PoincareBirkhoffWittBasis.element_class
            The element to display
        notebook : bool (optional, default: `True`)
            Uses IPython display with math if `True`, otherwise just returns the LaTeX code as string.
        """
        map_string = []
        first_term = True
        for monomial, coefficient in f.monomial_coefficients().items():
            alphas = monomial.dict().items()

            if first_term:
                if coefficient < 0:
                    sign_string = "-"
                else:
                    sign_string = ""
                first_term = False
            else:
                if coefficient < 0:
                    sign_string = "-"
                else:
                    sign_string = "+"
            if abs(coefficient) == 1:
                coeff_string = ""
            else:
                coeff_string = str(abs(coefficient))
            term_strings = [sign_string + coeff_string]
            for alpha, power in alphas:
                if power > 1:
                    power_string = r"^{" + str(power) + r"}"
                else:
                    power_string = ""
                alpha_string = "".join(
                    str(k) * int(-v) for k, v in alpha.monomial_coefficients().items()
                )
                term_strings.append(r"f_{" + alpha_string + r"}" + power_string)

            map_string.append(r"\,".join(term_strings))
        if notebook:
            display(Math(" ".join(map_string)))
        else:
            return " ".join(map_string)

    def _display_map(self, arrow, f):
        """Display a single arrow plus map of the BGG complex."""
        f_string = self.display_pbw(f, notebook=False)
        display(Math(r"\to".join(arrow) + r",\,\," + f_string))

    def display_maps(self, mu):
        """Display all the maps of the BGG complex for a given mu, in appropriate order.
        
        Parameters
        ----------
        mu : tuple
            tuple encoding the weight as linear combination of simple roots
        """
        maps = self.compute_maps(mu)
        maps = sorted(maps.items(), key=lambda s: (len(s[0][0]), s[0]))
        for arrow, f in maps:
            self._display_map(arrow, f)
