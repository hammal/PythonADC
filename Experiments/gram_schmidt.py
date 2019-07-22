import numpy as np


def gram_schmidt(v):
    print("Rank(V) = %i" % np.linalg.matrix_rank(v))
    u = np.zeros_like(v)
    u[0] = v[0]/np.linalg.norm(v[0],2)

    assert np.allclose(np.dot(u[0],u[0]),1)

    u_tmp = np.zeros_like(u[0])
    for k in range(1,len(v)):
        u_tmp = v[k] - sum(np.dot(v[k],u[i])*u[i] for i in range(k))
        u[k] = u_tmp/np.linalg.norm(u_tmp,2)

    assert np.allclose(np.dot(u.transpose(),u),np.eye(u.shape[0]))
    return u.transpose()



# A = np.random.randn(4,4)
# print("Rank(A) = %i " % np.linalg.matrix_rank(A))
# B = gram_schmidt(A)
# print("Rank(B) = %i " % np.linalg.matrix_rank(B))

# print("A^T * A = {}".format(np.dot(A.transpose(),A)))
# print("B^T * B = {}".format(np.dot(B.transpose(),B)))