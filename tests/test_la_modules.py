import pytest

try:
    import sage.all
except ImportError:
    import sage.all__sagemath_modules

from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix

from bggcohomology.bggcomplex import BGGComplex
from bggcohomology.la_modules import (
    BGGCohomology,
    LieAlgebraCompositeModule,
    ModuleFactory,
)
from bggcohomology.weight_set import WeightSet


@pytest.mark.parametrize("root_system", ["A2", "B2", "A3", "G2"])
def test_trivial_module(root_system):
    bgg = BGGComplex(root_system)
    factory = ModuleFactory(bgg.LA)
    component_dic = {"g": factory.build_component("g", "coad")}

    components = [[("g", 1, "wedge")]]
    module = LieAlgebraCompositeModule(factory, components, component_dic)
    assert bgg.LA.dimension() == module.total_dimension


@pytest.mark.parametrize("root_system", ["A2", "B2", "A3", "G2"])
@pytest.mark.parametrize("subalg", ["g", "b"])
def test_adjoint_action(root_system, subalg):
    bgg = BGGComplex(root_system)
    factory = ModuleFactory(bgg.LA)

    component = factory.build_component(subalg, "ad")
    for (i, j), Cijk in component.action.items():
        Li = factory.index_to_lie_algebra[i]
        Lj = factory.index_to_lie_algebra[j]
        Lij = Li.bracket(Lj)
        from_coeffs = sum(factory.index_to_lie_algebra[k] * c for k, c in Cijk.items())
        if from_coeffs - Lij != 0:
            raise AssertionError("Structure coefficients of action are incorrect")


@pytest.mark.parametrize("root_system", ["A2", "B2", "A3", "G2"])
@pytest.mark.parametrize("subset", [[], [1], [2], [1, 2]])
def test_adjoint_action_parabolic(root_system, subset):
    bgg = BGGComplex(root_system)
    factory = ModuleFactory(bgg.LA)

    component = factory.build_component("p", "ad", subset=subset)
    for (i, j), Cijk in component.action.items():
        Li = factory.index_to_lie_algebra[i]
        Lj = factory.index_to_lie_algebra[j]
        Lij = Li.bracket(Lj)
        from_coeffs = sum(factory.index_to_lie_algebra[k] * c for k, c in Cijk.items())
        if from_coeffs - Lij != 0:
            raise AssertionError("Structure coefficients of action are incorrect")


@pytest.mark.parametrize("root_system", ["A1", "A2", "B2", "A3", "G2"])
def test_cohom1(root_system):
    r""":math:`H^k(X,\mathfrak b)=0` for all k and for all types."""
    BGG = BGGComplex(root_system)
    factory = ModuleFactory(BGG.LA)
    component_dic = {"b": factory.build_component("b", "coad")}

    components = [[("b", 1, "wedge")]]
    module = LieAlgebraCompositeModule(factory, components, component_dic)

    bggcohom = BGGCohomology(BGG, module)
    all_degrees = range(BGG.max_word_length + 1)
    for i in all_degrees:
        assert bggcohom.betti_number(bggcohom.cohomology(i)) == 0


@pytest.mark.parametrize("root_system", ["A2", "A3", "A4"])
def test_cohom2(root_system):
    r""":math:`H^k(X,\mathfrak b\otimes \mathfrak u)=0` for all k>0 in type An, n>1"""
    BGG = BGGComplex(root_system)
    factory = ModuleFactory(BGG.LA)
    component_dic = {
        "b": factory.build_component("b", "coad"),
        "u": factory.build_component("u", "coad"),
    }

    components = [[("b", 1, "wedge"), ("u", 1, "wedge")]]
    module = LieAlgebraCompositeModule(factory, components, component_dic)

    cohom = BGGCohomology(BGG, module)
    bggcohom = BGGCohomology(BGG, module)
    assert bggcohom.betti_number(bggcohom.cohomology(0)) == 1
    all_degrees = range(1, BGG.max_word_length + 1)
    for i in all_degrees:
        assert bggcohom.betti_number(bggcohom.cohomology(i)) == 0


@pytest.mark.parametrize("root_system", ["A1", "A2", "B2", "A3", "G2"])
@pytest.mark.parametrize("action_type", ["ad", "coad"])
@pytest.mark.parametrize("subset", [[], [1]])
def test_coker(root_system, action_type, subset):
    """Use fact that coker(^2b->bxb) = sym^2b"""
    BGG = BGGComplex(root_system)
    factory = ModuleFactory(BGG.LA)

    component_dic = {"b": factory.build_component("b", action_type, subset=subset)}

    wedge_components = [[("b", 2, "wedge")]]
    wedge_module = LieAlgebraCompositeModule(factory, wedge_components, component_dic)

    tensor_components = [[("b", 1, "wedge"), ("b", 1, "wedge")]]
    tensor_module = LieAlgebraCompositeModule(factory, tensor_components, component_dic)

    sym_components = [[("b", 2, "sym")]]
    sym_module = LieAlgebraCompositeModule(factory, sym_components, component_dic)

    # Store cokernel in a dictionary
    # each key is a weight, each entry is a matrix encoding the basis of the cokernel
    T = dict()

    for mu in wedge_module.weight_components.keys():
        # Basis of the weight component mu of the wedge module
        wedge_basis = wedge_module.weight_components[mu][0][1]

        # Build the matrix as a sparse matrix
        sparse_mat = dict()

        for wedge_index, wedge_row in enumerate(wedge_basis):
            a, b = wedge_row  # each row consists of two indices

            # dictionary sending tuples of (a,b,0) to their index in the basis of tensor product module
            target_dic = tensor_module.weight_comp_index_numbers[mu]

            # look up index of a\otimes b and b\otimes a, and assign respective signs +1, -1
            index_1 = target_dic[(a, b, 0)]
            index_2 = target_dic[(b, a, 0)]
            sparse_mat[(wedge_index, index_1)] = 1
            sparse_mat[(wedge_index, index_2)] = -1

        # Build a matrix from these relations
        M = matrix(
            ZZ,
            sparse_mat,
            nrows=wedge_module.dimensions[mu],
            ncols=tensor_module.dimensions[mu],
            sparse=True,
        )

        # Cokernel is kernel of transpose
        T[mu] = M.transpose().kernel().basis_matrix()

    bggcohom1 = BGGCohomology(BGG, tensor_module, coker=T)
    bggcohom2 = BGGCohomology(BGG, sym_module)
    all_degrees = range(BGG.max_word_length + 1)
    for i in all_degrees:
        betti1 = bggcohom1.betti_number(bggcohom1.cohomology(i))
        betti2 = bggcohom2.betti_number(bggcohom2.cohomology(i))
        assert betti1 == betti2
