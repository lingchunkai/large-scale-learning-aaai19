cimport cython
import cython
from libcpp.vector cimport vector
'''
idx_t = cython.typedef(long)
VVI = cython.typedef(vector[vector[idx_t]]) 
VI = cython.typedef(vector[idx_t])
VD = cython.typedef(vector[double])
'''

ctypedef long idx_t
ctypedef vector[vector[idx_t]] VVI
ctypedef vector[idx_t] VI
ctypedef vector[double] VD

cdef VVI TransformLL(LL)
cdef VI TransformL(L)

cdef cppclass GameStructure:
    VVI u_c_I
    VVI v_c_I
    VVI u_c_A
    VVI v_c_A

    VI u_p_I
    VI v_p_I
    VI u_p_A
    VI v_p_A

    idx_t usize
    idx_t vsize

cdef class pywrap_GameStructure(object):
    cdef GameStructure *thisptr

