# -*- coding: utf8 -*-

"""
Deprecated. fast_module.py provides similar functionality (except for support of quotients),
but with much higher performance.
"""

from collections import defaultdict
from sympy.utilities.iterables import subsets
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.combinat.free_module import CombinatorialFreeModule
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.rings.rational_field import RationalField
from sage.algebras.lie_algebras.subalgebra import LieSubalgebra_finite_dimensional_with_basis
from sage.sets.family import Family
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ

class LieAlgebraModuleElement(IndexedFreeModuleElement):
    """Element class of LieAlgebraModule. We only modify __repr__ here for clear display"""

    def __init__(self, parent, x):
        super(LieAlgebraModuleElement, self).__init__(parent, x)

    def __repr__(self):
        if len(self.monomials()) == 0:
            return '0'
        output_string = ''

        # display all the monomials+coefficients as a sum, omitting + if there is a minus in the next term,
        # and omitting the coefficient if it is 1.
        for t, c in self.monomial_coefficients().items():
            if c < 0:
                output_string += '-'
                c *= -1
            elif len(output_string) > 0:
                output_string += '+'
            if c not in {1, -1}:
                output_string += str(c) + '*'
            output_string += str(t)
        return output_string


class LieAlgebraModule(CombinatorialFreeModule):
    """Class for working with Lie algebra modules (with basis). This class is designed to work well with modules over
    a Lie algebra G which are formed by the following pieces:
    - The Lie algebra G with (co)adjoint action
    - The positive/negative parts in the Chevallay basis plus the maximal torus, again with (co)adjoint action
    With operations, direct sum, tensor products, symmetric powers, exterior powers and quotients by submodules."""

    Element = LieAlgebraModuleElement

    @staticmethod
    def __classcall_private__(cls, base_ring, basis_keys=None, lie_algebra=None, action=None, **kwargs):
        if isinstance(basis_keys, (list, tuple)):
            basis_keys = FiniteEnumeratedSet(basis_keys)
        return super(LieAlgebraModule, cls).__classcall__(cls, base_ring, basis_keys=basis_keys, lie_algebra=lie_algebra, action=action, **kwargs)

    def __init__(self, base_ring, basis_keys=None, lie_algebra=None, action=None):
        self.lie_algebra = lie_algebra
        self._index_action = action
        self.basis_keys = basis_keys
        if isinstance(lie_algebra, LieSubalgebra_finite_dimensional_with_basis):
            self.ambient_lie_algebra = lie_algebra.ambient()
            ambient_basis = dict(self.ambient_lie_algebra.basis()).items()
            self.lie_algebra_basis = Family({k:v for k,v in ambient_basis if v in lie_algebra.basis()})
        else:
            self.ambient_lie_algebra = lie_algebra
            self.lie_algebra_basis = self.lie_algebra.basis()
        super(LieAlgebraModule, self).__init__(base_ring, basis_keys=basis_keys)

    def action(self, X, m):
        """self._index_action(X,m) produces a dictionary encoding the action on a basis element. This function
        uses self._index_action to extend to define the action of X on m, resulting in an element X.m in the
        module."""

        total = self.zero()
        for i, c in m.monomial_coefficients().iteritems():
            total += c * sum([d * self.basis()[j] for j, d in self._index_action(X, i).items()],
                             self.zero())
        return total

    def pbw_action(self, pbw_elt, m):
        """Repeatedly apply action to terms in a pbw polynomial to compute the action of the universal
        enveloping algebra on the Lie algebra"""
        total = self.zero()
        for monomial, coefficient in pbw_elt.monomial_coefficients().items():
            # convert monomials to list of roots, which are used as keys for the lie algebra basis
            # we reverse the final list because we act on the left.
            lie_alg_elements = [self.ambient_lie_algebra.basis()[term] for term in monomial.to_word_list()][::-1]
            sub_total = m
            for X in lie_alg_elements:
                sub_total = self.action(X, sub_total)
                if sub_total == self.zero():
                    break
            total += coefficient*sub_total
        return total

    def direct_sum(*modules):
        """Given a list of modules, produces a new module encoding the direct sum of these modules. The basis is the
        join of the bases of the modules. We use the class DirectSum() to keep track of which basis elements belongs
        to which original module. The action on the direct sum of modules is defined by the restriction on the original
        modules on each module."""

        new_basis = []
        actions = dict()
        for i, module in enumerate(modules):
            new_basis += [DirectSum(i, key) for key in module.basis_keys]
            actions[i] = module._index_action

        def action(X, m):
            return direct_sum_map(m.index, actions[m.index](X, m.key))

        return LieAlgebraModule(modules[0].base_ring(), new_basis, modules[0].lie_algebra, action)

    def tensor_product(*modules):
        """Gives the tensor product of the modules. The basis is given by the set of all tuples, such that each
        element in the tuple belongs to precisely one of the original modules. The action is defined through
        the coproduct, e.g. X.(a (x) b) = (X.a)(x)b+a(x)(X.b)."""

        for module in modules:
            if len(module.basis_keys) == 0:
                return LieAlgebraModule(modules[0].base_ring(), [], modules[0].lie_algebra, lambda X, k: {})
        #if len([module for module in modules if len(module.basis_keys) > 1]) > 0:
        #    modules = [module for module in modules if len(module.basis_keys) > 1]
        #else:
        #    return LieAlgebraModule(modules[0].base_ring(), [1], modules[0].lie_algebra, lambda X, k: {})

        new_basis = LieAlgebraModule._tensor_product_basis(*[module.basis_keys for module in modules])

        def action(X, m):
            out_dict = defaultdict(int)
            for index, key in enumerate(m):
                action_on_term = modules[index]._index_action(X, key)
                #index_dict = Counter({m.replace(index, t): c for t, c in action_on_term.items()})
                #out_dict += index_dict
                for t, c in action_on_term.items():
                    out_dict[m.replace(index, t)] += c
            return out_dict
        return LieAlgebraModule(modules[0].base_ring(), new_basis, modules[0].lie_algebra, action)

    def symmetric_power(self, n):
        """Gives n-fold symmetric power of module. The basis is given by the set of all sub-multisets (with repetition)
        of size n of the basis of the module. The action is induced from the action on the tensor product.
        In the special case that n=0, we return the module spanned by 0 elements. Technically we should return the
        base ring, but I first need to program the base ring as a Lie algebra module. For n==1 we return a fresh
        instance of the same LieAlgebraModule."""

        if n == 0:
            return LieAlgebraModule(self.base_ring(), [1], self.lie_algebra, lambda X, k: dict())
        if n == 1:
            return LieAlgebraModule(self.base_ring(), self.basis_keys, self.lie_algebra, self._index_action)

        new_basis = subsets(self.basis_keys, n, repetition=True)
        new_basis = [SymmetricProduct(*i) for i in new_basis]

        def action(X, m):
            out_dict = defaultdict(int)
            for index, key in enumerate(m):
                action_on_term = self._index_action(X, key)
                #index_dict = Counter({m.replace(index, t): c for t, c in action_on_term.items()})
                #out_dict += index_dict
                for t, c in action_on_term.items():
                    out_dict[m.replace(index, t)] += c
            return out_dict

        return LieAlgebraModule(self.base_ring(), new_basis, self.lie_algebra, action)

    def alternating_power(self, n):
        """Gives n-fold symmetric power of module. The basis is given by the set of all subsets (without repetition)
        of size n of the basis of the module. The action is induced from the action on the tensor product. In the
        special case that n=0, we return the free module on zero generators. Technically we should return the base
        ring, but I first need to program the base ring as a Lie algebra module. For n==1 we return a fresh instance
        of the same LieAlgebraModule. If n is larger than the dimension of the module,we also return the free module
        on zero generators."""

        if n == 0:
            return LieAlgebraModule(self.base_ring(), [1], self.lie_algebra, lambda X, k: {})
        if n == 1:
            return LieAlgebraModule(self.base_ring(), self.basis_keys, self.lie_algebra, self._index_action)
        if n > len(self.basis_keys):
            return LieAlgebraModule(self.base_ring(), [], self.lie_algebra, lambda X, k: {})

        new_basis = subsets(self.basis_keys, n, repetition=False)
        new_basis = [AlternatingProduct(*i) for i in new_basis]

        def action(X, m):
            out_dict = defaultdict(int)
            for index, key in enumerate(m):
                action_on_term = self._index_action(X, key)
                for t, c in action_on_term.items():
                    unsorted_term = m.replace(index, t)
                    parity = unsorted_term.parity()
                    if parity != 0:
                        out_dict[unsorted_term.sort()] += c * parity
            return out_dict

        return LieAlgebraModule(self.base_ring(), new_basis, self.lie_algebra, action)

    @staticmethod
    def _tensor_product_basis(*iterators):
        """gives the set of all (ordered) tuples of same length as number of iterators, where the ith element belongs
        to the ith iterator. """
        iterators = list(iterators)
        basis = [[x] for x in iterators.pop()]
        while len(iterators) > 0:
            iterator = iterators.pop()
            basis = [[x] + y for x in iterator for y in basis]
        basis = [TensorProduct(*b) for b in basis]
        return basis


class DirectSum(object):
    def __init__(self, index, key):
        self.index = int(index)
        self.key = key

    def __hash__(self):
        return hash((self.index, self.key))

    def __eq__(self, other):
        return type(self)==type(other) and (self.key, self.index) == (other.key, other.index)

    def __repr__(self):
        return str(self.key)


def direct_sum_map(index, dic):
    return {DirectSum(index, k): v for k, v in dic.items()}


class TensorProduct(object):
    def __init__(self, *keys):
        self.keys = list(keys)

    def __hash__(self):
        return hash(tuple(self.keys))

    def __eq__(self, other):
        return type(self)==type(other) and self.keys == other.keys

    def __repr__(self):
        return '⊗'.join([str(k) for k in self.keys])

    def __getitem__(self, index):
        return self.keys[index]

    def __setitem__(self, index, value):
        self.keys[index] = value
        return self

    def replace(self, index, value):
        keys = self.keys[:]
        keys[index] = value
        return TensorProduct(*keys)

    def insert(self, index, value):
        keys = self.keys[:]
        keys.insert(index, value)
        return TensorProduct(*keys)


class SymmetricProduct(object):
    def __init__(self, *keys):
        self.keys = sorted(keys)

    def __hash__(self):
        return hash(tuple(self.keys))

    def __eq__(self, other):
        return type(self)==type(other) and self.keys == other.keys

    def __repr__(self):
        return '⊙'.join([str(k) for k in self.keys])

    def __getitem__(self, index):
        return self.keys[index]

    def __setitem__(self, index, value):
        self.keys[index] = value
        return self

    def replace(self, index, value):
        keys = self.keys[:]
        keys[index] = value
        return SymmetricProduct(*keys)

    def insert(self, value):
        keys = self.keys[:]
        keys.append(value)
        return SymmetricProduct(*keys)


class AlternatingProduct(object):
    def __init__(self, *keys):
        self.keys = list(keys)

    def __hash__(self):
        return hash(tuple(self.keys))

    def __eq__(self, other):
        return type(self)==type(other) and self.keys == other.keys

    def __repr__(self):
        return '∧'.join([str(k) for k in self.keys])

    def __getitem__(self, index):
        return self.keys[index]

    def __setitem__(self, index, value):
        self.keys[index] = value
        return self

    def replace(self, index, value):
        keys = self.keys[:]
        keys[index] = value
        return AlternatingProduct(*keys)

    def insert(self, value, index=0):
        keys = self.keys[:]
        keys.insert(index, value)
        return AlternatingProduct(*keys)

    def parity(self):
        """Computes the parity of the permutation. """
        a = self.keys[:]
        if len(set(a)) < len(a):
            return 0
        b = sorted(a)
        inversions = 0
        while a:
            first = a.pop(0)
            inversions += b.index(first)
            b.remove(first)
        return -1 if inversions % 2 else 1

    def sort(self):
        self.keys = sorted(self.keys)
        return self


class LieAlgebraModuleFactory:
    """s"""
    def __init__(self, lie_algebra):
        self.lie_algebra = lie_algebra
        self.lattice = lie_algebra.weyl_group().domain().root_system.root_lattice()
        self.lie_algebra_basis = self.lie_algebra.basis()

        self.f_roots = list(self.lattice.negative_roots())
        self.e_roots = list(self.lattice.positive_roots())
        self.h_roots = self.lattice.alphacheck().values()

        self._initialize_root_dictionary()

        self.basis = dict()
        self.basis['g'] = sorted(self.string_to_root.keys())
        self.basis['u'] = sorted([self.root_to_string[r] for r in self.e_roots])
        self.basis['n'] = sorted([self.root_to_string[r] for r in self.f_roots])
        self.basis['b'] = sorted(self.basis['n'] + [self.root_to_string[r] for r in self.h_roots])

        self._bracket_dict = dict()
        for X in self.lie_algebra_basis:
            for Y in self.lie_algebra_basis:
                self._bracket_dict[(X,Y)]=lie_algebra.bracket(X,Y)

        self._coadjoint_dict_cache = dict()

        self.subalgebra = dict()
        self.subalgebra['g'] = self.lie_algebra
        self.subalgebra['u'] = self._basis_to_subalgebra(self.basis['u'])
        self.subalgebra['n'] = self._basis_to_subalgebra(self.basis['n'])
        self.subalgebra['b'] = self._basis_to_subalgebra(self.basis['b'])

        self.dual_root_dict = self._init_dual_root_dict()

    def parabolic_p_module(self, subset, base_ring=RationalField()):
        e_roots_in_span = [r for r in self.e_roots if set(r.monomial_coefficients().keys()).issubset(subset)]
        basis = self.basis['b']+sorted([self.root_to_string[r] for r in e_roots_in_span])
        subalgebra = self._basis_to_subalgebra(basis)
        return basis, LieAlgebraModule(base_ring, basis, subalgebra, self.fast_adjoint_action)

    def parabolic_n_module(self, subset, base_ring=RationalField()):
        f_roots_not_in_span = [r for r in self.f_roots if not set(r.monomial_coefficients().keys()).issubset(subset)]
        basis = [self.root_to_string[r] for r in f_roots_not_in_span]
        basis = sorted(basis)
        subalgebra = self._basis_to_subalgebra(basis)
        return  basis, LieAlgebraModule(base_ring, basis, subalgebra, self.fast_adjoint_action)

    def parabolic_u_module(self, subset, base_ring=RationalField()):
        e_roots_not_in_span = [r for r in self.e_roots if not set(r.monomial_coefficients().keys()).issubset(subset)]
        basis = [self.root_to_string[r] for r in e_roots_not_in_span]
        basis = sorted(basis)
        subalgebra = self._basis_to_subalgebra(basis)
        key = tuple(['u',tuple(subset)])
        self.compute_coadjoint_dict(basis,key)
        coadjoint_action = lambda X, m: self.fast_coadjoint_action(X, m, key)
        return  basis, LieAlgebraModule(base_ring, basis, subalgebra, coadjoint_action)

    def _initialize_root_dictionary(self):
        def root_dict_to_string(root_dict):
            return ''.join(''.join([str(k)] * abs(v)) for k, v in root_dict.items())

        self.string_to_root = dict()
        for i, b in dict(self.lattice.alphacheck()).items():
            self.string_to_root['h%d' % i] = b
        for a in self.lattice.negative_roots():
            key = 'f' + root_dict_to_string(a.monomial_coefficients())
            self.string_to_root[key] = a
        for a in self.lattice.positive_roots():
            key = 'e' + root_dict_to_string(a.monomial_coefficients())
            self.string_to_root[key] = a

        self.root_to_string = {r: i for i, r in self.string_to_root.items()}

    def string_to_lie_algebra(self, m):
        return  self.lie_algebra_basis[self.string_to_root[m]]

    def _basis_to_subalgebra(self, basis):
        basis = [self.lie_algebra_basis[self.string_to_root[r]] for r in basis]
        return self.lie_algebra.subalgebra(basis)

    def lie_alg_to_module_basis(self, X):
        """Takes an element of the Lie algebra and writes it out as a dict in the module basis"""
        out_dict = defaultdict(int)
        for t, c in X.monomial_coefficients().items():
            out_dict[self.root_to_string[t]] = c
        return out_dict

    def adjoint_action(self, X, m):
        """Takes X and element of the Lie algebra, and m an index of the basis of the Lie algebra, and outputs
        the adjoint action of X on the corresponding basis element"""
        bracket = self.lie_algebra.bracket(X, self.string_to_lie_algebra(m))
        return self.lie_alg_to_module_basis(bracket)

    def fast_adjoint_action(self,X,m):
        output = self.lie_algebra.zero()
        Y = self.string_to_lie_algebra(m)
        try:
            for monomial,coefficient in X.monomial_coefficients().items():
                output+=coefficient*self._bracket_dict[(self.lie_algebra_basis[monomial],Y)]
        except Exception as error:
            print('these guys gave an error: %s, %s, %s' % (X,Y,m))
            print('monomial_coefficients is: %s ' % X.monomial_coefficients())
            print('type is %s' % type(X))
            raise error
        return self.lie_alg_to_module_basis(output)

    def _init_dual_root_dict(self):
        """"Create dictionary mapping roots to their dual"""
        dual_root_dict = dict()
        for root in self.e_roots + self.f_roots:
            dual_root_dict[self.root_to_string[-root]] = self.root_to_string[root]
        for root in self.h_roots:
            dual_root_dict[self.root_to_string[root]] = self.root_to_string[root]
        return dual_root_dict

    def pairing(self, X, Y):
        """Natural pairing in the Chevallay basis. It is defined by (e_I,f_I)=1, (h_i,h_i)=1, and 0 otherwise"""
        return sum(c1 * c2 for x1, c1 in X.items() for x2, c2 in Y.items() if x1 == self.dual_root_dict[x2])

    def coadjoint_action(self, X, m, basis):
        """ Uses pairing and adjoint action to define coadjoint action corestricted to a subspace defined by `basis`.
         The coadjoint action is defined by
        coad(X)(m) = - sum( pairing(m,ad(X, dual(alpha))*alpha for alpha in basis)"""
        output = dict()  # Output is a dictionary of monomials
        for alpha in basis:
            alpha_dual = self.string_to_lie_algebra(self.dual_root_dict[alpha])
            bracket = self.lie_algebra.bracket(X, alpha_dual)
            bracket = self.lie_alg_to_module_basis(bracket)  # Compute bracket and convert to dictionary of monomials
            inn_product = self.pairing(bracket, {m: 1})
            if inn_product != 0: 
                output[alpha] = -inn_product  # If the action on alpha is non-zero, add it to the output dictionary
        return output

    def compute_coadjoint_dict(self,basis,key):
        """Precomputes the coadjoint action on a basis corestricted to a subalgebra, and caches the results.
        Returns a function that takes in X,m, like coadjoint_action or adjoint_action"""
        if key in self._coadjoint_dict_cache:
            return self._coadjoint_dict_cache[key]
        else:
            coadjoint_dict=dict()
            for X in self.lie_algebra_basis:
                for Y in self.lie_algebra_basis:
                    coadjoint_bracket = self.lie_algebra.zero()
                    m = self.lie_alg_to_module_basis(Y)
                    for alpha in basis:
                        alpha_dual = self.string_to_lie_algebra(self.dual_root_dict[alpha])
                        bracket = self.lie_algebra.bracket(X, alpha_dual)
                        bracket = self.lie_alg_to_module_basis(bracket)
                        inn_product = self.pairing(bracket, m)
                        if inn_product != 0:
                            coadjoint_bracket += -inn_product*self.string_to_lie_algebra(alpha)
                    coadjoint_dict[(X,Y)]=coadjoint_bracket
            self._coadjoint_dict_cache[key] = coadjoint_dict
            return coadjoint_dict

    def fast_coadjoint_action(self,X,m,key):
        output = self.lie_algebra.zero()
        Y = self.string_to_lie_algebra(m)
        for monomial,coefficient in X.monomial_coefficients().items():
            output+=coefficient*self._coadjoint_dict_cache[key][(self.lie_algebra_basis[monomial], Y)]
        return self.lie_alg_to_module_basis(output)


    def construct_module(self, base_ring=RationalField(), subalgebra='g', action='adjoint'):
        """Return a LieAlgebraModule with basis some subalgebra of g, and with either adjoint or coadjoint action.
        These form the 'building blocks' for more complicated modules."""
        action_map = {'adjoint':self.fast_adjoint_action,
                      'coadjoint': (lambda X,m: self.fast_coadjoint_action(X, m, subalgebra))}
        return LieAlgebraModule(base_ring, self.basis[subalgebra], self.subalgebra[subalgebra], action_map[action])


class WeightModuleWithRelations(LieAlgebraModule):
    """Class for taking care of weight modules with relations. The class constructor takes
    a LieAlgebraModule together with a map that gives a 'weight' for each basis key (in principle
    the weights can be any hashable object). The class generates a list weight_dic sending weights to
    all the basis elements with this weight.

    The class also takes an optional third argument
    which is a list with dictionaries of form {..., key:integer, ...}, where key has to be
    in the basis of the LieAlgebraModule. These are the relations, and we assume that each element of the
    list of relations has a well-defined weight. The class generates a dictionary relations_weight_dic
    containing all the relations for a specified weight, and for each weight it defines a section from
    the quotient module M/R -> M. """

    @staticmethod
    def __classcall_private__(cls, base_ring, lie_alg_module=None, get_weight=None, relations=None, **kwargs):
        if isinstance(relations, (list, tuple)):  # ensure that the relations argument is hashable
            relations = FiniteEnumeratedSet([tuple(r.items()) for r in relations])

        basis_keys = lie_alg_module.basis_keys
        lie_algebra = lie_alg_module.lie_algebra
        action = lie_alg_module.action

        return super(WeightModuleWithRelations, cls).__classcall__(cls, base_ring, basis_keys=basis_keys,
                                                                   lie_algebra=lie_algebra, action=action,
                                                                   lie_alg_module=lie_alg_module,
                                                                   get_weight=get_weight,
                                                                   relations=relations, **kwargs
                                                                   )

    def __init__(self, base_ring, lie_alg_module, get_weight, relations=None, **kwargs):
        self.get_weight = get_weight
        if relations is not None:  # turn relations back from a FiniteEnumeratedSet of tuples into a dictionary
            self.relations = [{x: y for x, y in r} for r in relations]
        else:
            self.relations = []

        super(WeightModuleWithRelations, self).__init__(base_ring, lie_alg_module.basis_keys,
                                                        lie_alg_module.lie_algebra, lie_alg_module._index_action)

        # Initialize dictionary of weights
        self.weight_dic = dict()
        for key in self.basis_keys:
            weight = self.get_weight(key)
            if weight not in self.weight_dic:
                self.weight_dic[weight] = [key]
            else:
                self.weight_dic[weight] += [key]

        # Initialize dictionary of weights for the relations
        self.relations_weight_dic = {weight: [] for weight in self.weight_dic.keys()}
        for dic in self.relations:
            weight = self.get_weight(dic.keys()[0])
            self.relations_weight_dic[weight] += [dic]

        # Initialize the cache of sections from the quotient module
        self._sections = dict()

    @staticmethod
    def vectorize_relations(keys, relations):
        """Given a set of keys and relations, return a matrix encoding the (transpose of the)
         relations in the basis given by the keys"""
        output = matrix(ZZ, len(keys), len(relations), 0)
        index_dict = {key:index for index,key in enumerate(keys)}
        #for index, key in enumerate(keys):
        #    index_dict[key] = index
        for row, dic in enumerate(relations):
            for key, value in dic.items():
                output[index_dict[key], row] = value
        return output

    def get_section(self,weight):
        """Compute a section of the quotient module for the given weight. The section is computed by
        computing the kernel of the transpose of the matrix defining the relations."""
        # Use cached result if available
        if weight in self._sections:
            return self._sections[weight]

        # If the weight is not in the module, section is automatically trivial
        if weight not in self.weight_dic:
            return []

        # If there are no relations for this weight, section is just the original weight space.
        if weight not in self.relations_weight_dic:
            return self.weight_dic[weight]

        relations = self.relations_weight_dic[weight]
        keys = self.weight_dic[weight]
        vectorized_relations = self.vectorize_relations(keys, relations)
        section = []

        # Compute kernel of transpose of relations, and turn the result back into a dictionary
        for row in vectorized_relations.left_kernel().basis():
            section.append({k: c for k, c in list(zip(keys, row)) if c != 0})

        # Cache the result
        self._sections[weight] = section

        return section
