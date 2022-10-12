#cython: boundscheck=True, wraparound=False, nonecheck=False, cdivision=True, profile=False, linetrace=False
cimport numpy as np
import numpy as np
cimport cython

cdef VVI TransformLL(LL):
    ''' Transform list of lists into vector of vectors'''
    cdef VVI ret
    for L in LL:
        ret.push_back(VI())
        for i in L:
            ret.back().push_back(i)

    return ret

cdef VI TransformL(L):
    ''' Transform list into vector '''
    cdef VI ret
    for i in L:
        if i is None: ret.push_back(-1)
        else: ret.push_back(i)
    return ret

cdef class pywrap_GameStructure(object):
    def __init__(self):
        self.thisptr = new GameStructure()

def MakeGameStructure(uv_c_I, uv_c_A, uv_p_I, uv_p_A):
    ret = pywrap_GameStructure()
    # cdef GameStructure ret()
    ret.thisptr.u_c_I = TransformLL(uv_c_I[0])
    ret.thisptr.v_c_I = TransformLL(uv_c_I[1])
    ret.thisptr.u_c_A = TransformLL(uv_c_A[0])
    ret.thisptr.v_c_A = TransformLL(uv_c_A[1])

    ret.thisptr.u_p_I = TransformL(uv_p_I[0])
    ret.thisptr.v_p_I = TransformL(uv_p_I[1])
    ret.thisptr.u_p_A = TransformL(uv_p_A[0])
    ret.thisptr.v_p_A = TransformL(uv_p_A[1])

    ret.thisptr.usize = ret.thisptr.u_p_A.size()
    ret.thisptr.vsize = ret.thisptr.v_p_A.size()

    # print(ret.thisptr.v_c_I)

    return ret
