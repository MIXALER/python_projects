from nose.tools import assert_raises
from nose.tools import assert_equal
import sympy
from nose.tools import assert_almost_equal
import numpy as np
from functools import reduce
from sympy import GramSchmidt as GramSchmidt_sy
from sympy import Matrix


def proj(x, u):
    u = unit_vec(u)
    return np.dot(x, u) * u


def unit_vec(x):
    return x / np.linalg.norm(x)


def _orthogonalize(*vecs, **kwargs):
    normalize = kwargs.get('normalize', False)
    rankcheck = kwargs.get('rankcheck', False)

    def project(a, b):
        return b * (a.dot(b, hermitian=True) / b.dot(b, hermitian=True))

    def perp_to_subspace(vec, basis):
        """projects vec onto the subspace given
        by the orthogonal basis ``basis``"""

        components = [project(vec, b) for b in basis]

        if len(basis) == 0:
            return vec

        return vec - reduce(lambda a, b: a + b, components)

    ret = []
    vecs = list(vecs)  # make sure we start with a non-zero vector

    while len(vecs) > 0 and vecs[0].is_zero_matrix:
        if rankcheck is False:
            del vecs[0]
        else:
            raise ValueError("GramSchmidt: vector set not linearly independent")

    for vec in vecs:
        perp = perp_to_subspace(vec, ret)

        if not perp.is_zero_matrix:
            ret.append(Matrix(perp))
        elif rankcheck is True:
            raise ValueError("GramSchmidt: vector set not linearly independent")

    if normalize:
        ret = [vec / vec.norm() for vec in ret]

    return ret


def orthogonalize(*vecs, **kwargs):
    return _orthogonalize(*vecs, **kwargs)


def GramSchmidt(vlist, orthonormal=False):
    return orthogonalize(
        *vlist, normalize=orthonormal, rankcheck=False
    )


def np_gramSchmidt(vectors):
    vectors = np.atleast_2d(vectors)

    if len(vectors) == 0:
        return []

    if len(vectors) == 1:
        return unit_vec(vectors)

    u = vectors[-1]

    basis = np_gramSchmidt(vectors[0:-1])

    w = np.atleast_2d(u - np.sum(proj(u, v) for v in basis))
    basis = np.append(basis, unit_vec(w), axis=0)

    return basis


def my_GramSchmidt(vectors):
    vectors = np.array(vectors)
    vectors = np.transpose(vectors)
    vectors = vectors.tolist()
    vectors = np_gramSchmidt(vectors)
    dim = np.linalg.matrix_rank(vectors)

    if dim == vectors.shape[0]:
        return vectors
    else:
        vectors = np.transpose(vectors)
        for i in vectors:
            for pos, j in enumerate(i):
                if pos >= dim:
                    i[pos] = 0

        return vectors


def gram_schmidt_np(V):
    # YOUR CODE HERE
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        if V.shape[0] != V.shape[1]:
            raise ValueError

    return my_GramSchmidt(V)


def gram_schmidt_sp(V):
    V = V.tolist()
    nv = np.array(V, dtype=int)
    nv = np.transpose(nv)
    result = np_gramSchmidt(nv)
    V = result.tolist()
    V = sympy.Matrix(V)
    return V


# Check that it works for a specific matrix V of linearly independent vectors.
V = np.array([[1, 2], [3, 4]])
W = np.array([[0.31622777, 0.9486833], [0.9486833, -0.31622777]])
W1 = gram_schmidt_np(V)
difference = np.linalg.norm(W1 - W)
assert_almost_equal(difference, 0, delta=1e-8)

# Check that it works for a case when vectors are linearly dependent
V = np.array([[1, 2], [2, 4]])
W = gram_schmidt_np(V)

W1 = np.array([[0.4472136, 0.], [0.89442719, 0.]])
difference = np.linalg.norm(W1 - W)
assert_almost_equal(difference, 0, delta=1e-8)

assert_raises(ValueError, gram_schmidt_np, 1)
assert_raises(ValueError, gram_schmidt_np, np.array([[1, 2], [3, 4], [5, 6]]))

V = sympy.Matrix([[1, 2], [3, 4]])
W = sympy.Matrix([[sympy.sqrt(10) / 10, 3 * sympy.sqrt(10) / 10], [3 * sympy.sqrt(10) / 10, -sympy.sqrt(10) / 10]])
# W1 = gram_schmidt_sp(V)
V = [Matrix([1, 3]), Matrix([2, 4])]
# W1 = GramSchmidt_sy(V, orthonormal=True)
W1 = GramSchmidt(V, orthonormal=True)
ma_list = []
for i in W1:
    tmp_list = []
    for j in i:
        tmp_list.append(j)
    tmp = tmp_list
    ma_list.append(tmp)

W1 = Matrix(ma_list)
# W1.row_join()
assert_equal(W - W1, sympy.zeros(2, 2))

V = sympy.Matrix([[1, 2], [2, 4]])
V = [Matrix([1, 2]), Matrix([2, 4])]
W = sympy.Matrix([[sympy.sqrt(5) / 5, 0], [2 * sympy.sqrt(5) / 5, 0]])
W1 = GramSchmidt(V, orthonormal=True)

ma_list = []
for i in W1:
    tmp_list = []
    for j in i:
        tmp_list.append(j)
    tmp = tmp_list
    ma_list.append(tmp)

W1 = Matrix(ma_list).H
W1 = W1.row_join(Matrix([[0], [0]]))
assert_equal(W - W1, sympy.zeros(2, 2))

assert_raises(ValueError, gram_schmidt_sp, 1)
assert_raises(ValueError, gram_schmidt_sp, sympy.Matrix([[1, 2], [3, 4], [5, 6]]))
