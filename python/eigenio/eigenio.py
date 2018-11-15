"""
Binary Eigen Matrices IO
"""
from enum import Enum, IntEnum
import numpy as np
import scipy.sparse as sp
import sys

ONE_BYTE = 1
INT32_BYTES = 4

class MatrixFormat(IntEnum):
    MATRIX_TYPE=0
    SCALAR_TYPE=1


class MatrixType(Enum):
    DENSE_MATRIX=False
    SPARSE_MATRIX=True


class ScalarType(Enum):
    INTEGRAL=False
    FLOATING_POINT=True

    def from_kind(k):
        if k == 'i':
            return ScalarType.INTEGRAL
        elif k == 'f':
            return ScalarType.FLOATING_POINT
        else:
            raise ValueError("Unknown scalar type '{}'".format(k))


def set_bit(val, num, bitval):
    val = (val & ~(1<<num)) | (bitval << num)
    return val


def get_bit(val, num):
    return (val & (1 << num)) != 0


class CommonHeader:
    SIZE_BYTES = 2
    MATRIX_FORMAT = 0
    SCALAR_SIZE = 1
    
    def __init__(self, header_bytes=None, mtype=None, stype=None, ssize=None):
        if header_bytes is None:
            header_bytes = bytearray(CommonHeader.SIZE_BYTES)
        self.header_bytes = header_bytes
        self.matrix_type(mtype)
        self.scalar_type(stype)
        self.scalar_size(ssize)

    def matrix_type(self, mtype=None):
        if mtype is not None:
            self.header_bytes[CommonHeader.MATRIX_FORMAT] = set_bit(
                self.header_bytes[CommonHeader.MATRIX_FORMAT],
                MatrixFormat.MATRIX_TYPE,
                mtype.value)
        return MatrixType(get_bit(self.header_bytes[CommonHeader.MATRIX_FORMAT],
                                  MatrixFormat.MATRIX_TYPE))
        
    def scalar_type(self, stype=None):
        if stype is not None:
            self.header_bytes[CommonHeader.MATRIX_FORMAT] = set_bit(
                self.header_bytes[CommonHeader.MATRIX_FORMAT],
                MatrixFormat.SCALAR_TYPE,
                stype.value)
        return ScalarType(get_bit(self.header_bytes[CommonHeader.MATRIX_FORMAT],
                                  MatrixFormat.SCALAR_TYPE))

    def scalar_size(self, size=None):
        if size is not None:
            if size > 255: raise ValueError("Scalar size >255 (got {})".format(size))
            if size < 1: raise ValueError("Scalar size <1 (got {})".format(size))
            self.header_bytes[CommonHeader.SCALAR_SIZE] = size
        return self.header_bytes[CommonHeader.SCALAR_SIZE]


    def dtype(self):
        t = self.scalar_type()
        s = self.scalar_size()
        if t == ScalarType.INTEGRAL:
            t_c = 'i'
        else:
            t_c = 'f'

        return t_c + str(s)


class DenseHeader:
    SIZE_BYTES = 8
    ROWS = slice(0,4)
    COLS = slice(4,8)
    
    def __init__(self, common_header, header_bytes=None, rows=None, cols=None):
        self.common_header = common_header
        if(header_bytes is None):
            header_bytes = bytearray(DenseHeader.SIZE_BYTES)
        self.header_bytes = header_bytes
        self.rows(rows)
        self.cols(cols)

    def rows(self, rows=None):
        if rows is not None:
            self.header_bytes[DenseHeader.ROWS] = rows.to_bytes(INT32_BYTES,
                                                                byteorder=sys.byteorder)
        return int.from_bytes(self.header_bytes[DenseHeader.ROWS], byteorder=sys.byteorder)

    def cols(self, cols=None):
        if cols is not None:
            self.header_bytes[DenseHeader.COLS] = cols.to_bytes(INT32_BYTES,
                                                                byteorder=sys.byteorder)
        return int.from_bytes(self.header_bytes[DenseHeader.COLS], byteorder=sys.byteorder)

    def load_matrix(self, f):
        r = self.rows()
        c = self.cols()
        s = self.common_header.scalar_size()

        b = f.read(r * c * s)

        m = np.frombuffer(b, dtype=self.common_header.dtype())
        return np.reshape(m, (r,c), order='F')
    
    def store_matrix(self, m, f):
        f.write(self.common_header.header_bytes)
        f.write(self.header_bytes)
        b = m.tobytes(order='F')
        f.write(b)


class SparseHeader:
    SIZE_BYTES = 13
    SIS = slice(0,1)
    SNNZ = slice(1,5)
    SINNER = slice(5,9)
    SOUTER = slice(9,13)

    def __init__(self, common_header, header_bytes=None,
                 sis=None, snnz=None, sinner=None, souter=None):
        self.common_header = common_header
        if(header_bytes is None):
            header_bytes = bytearray(SparseHeader.SIZE_BYTES)
        self.header_bytes = header_bytes
        self.sis(sis)
        self.snnz(snnz)
        self.sinner(sinner)
        self.souter(souter)

    def sis(self, sis=None):
        if sis is not None:
            self.header_bytes[SparseHeader.SIS] = sis.to_bytes(ONE_BYTE,
                                                               byteorder=sys.byteorder)
        return int.from_bytes(self.header_bytes[SparseHeader.SIS], byteorder=sys.byteorder)

    def dtype_sis(self):
        return 'i'+str(self.sis())
    
    def snnz(self, snnz=None):
        if snnz is not None:
            self.header_bytes[SparseHeader.SNNZ] = snnz.to_bytes(INT32_BYTES,
                                                                byteorder=sys.byteorder)
        return int.from_bytes(self.header_bytes[SparseHeader.SNNZ], byteorder=sys.byteorder)
    
    def sinner(self, sinner=None):
        if sinner is not None:
            self.header_bytes[SparseHeader.SINNER] = sinner.to_bytes(INT32_BYTES,
                                                                byteorder=sys.byteorder)
        return int.from_bytes(self.header_bytes[SparseHeader.SINNER], byteorder=sys.byteorder)

    def souter(self, souter=None):
        if souter is not None:
            self.header_bytes[SparseHeader.SOUTER] = souter.to_bytes(INT32_BYTES,
                                                                byteorder=sys.byteorder)
        return int.from_bytes(self.header_bytes[SparseHeader.SOUTER], byteorder=sys.byteorder)

    def load_matrix(self, f):
        sis = self.sis()
        dtype_sys = self.dtype_sis()
        snnz = self.snnz()
        sinner = self.sinner()
        souter = self.souter()
        s = self.common_header.scalar_size()

        nnz_b = f.read(snnz * s)
        inner_b = f.read( snnz * sis)
        outer_b = f.read((souter + 1) * sis)

        nnz = np.frombuffer(nnz_b, dtype=self.common_header.dtype())
        inner = np.frombuffer(inner_b, dtype=dtype_sys)
        outer = np.frombuffer(outer_b, dtype=dtype_sys)
        
        m = sp.csc_matrix((nnz, inner, outer), shape=[sinner, souter])
        return m

    def store_matrix(self, m, f):
        f.write(self.common_header.header_bytes)
        f.write(self.header_bytes)
        
        b = m.data.tobytes(order='F')
        f.write(b)

        b = m.indices.tobytes(order='F')
        f.write(b)
        
        b = m.indptr.tobytes(order='F')
        f.write(b)

        
def load_stream(f):
    b = f.read(CommonHeader.SIZE_BYTES)
    common_header = CommonHeader(b)

    if common_header.matrix_type() == MatrixType.DENSE_MATRIX:
        header_cls = DenseHeader
    elif common_header.matrix_type() == MatrixType.SPARSE_MATRIX:
        header_cls = SparseHeader
    b = f.read(header_cls.SIZE_BYTES)
    header = header_cls(common_header, b)

    return header.load_matrix(f)


def store_stream(m, f):
    
    if sp.isspmatrix(m):
        matrix_type = MatrixType.SPARSE_MATRIX
        m = sp.csc_matrix(m)
    elif isinstance(m, np.ndarray):
        matrix_type = MatrixType.DENSE_MATRIX
    else:
        raise ValueError("Cannot store matrix of type {}".format(m.__class__))

    dtype = m.dtype
    scalar_size = dtype.itemsize
    scalar_type = ScalarType.from_kind(dtype.kind)
    ch = CommonHeader(None, matrix_type, scalar_type, scalar_size)

    shape = m.shape
    if matrix_type == MatrixType.SPARSE_MATRIX:
        sis = m.indices.dtype.itemsize
        snnz = m.nnz
        sinner = shape[0]
        souter = shape[1]
        h = SparseHeader(ch, None, sis, snnz, sinner, souter) 
    else:      
        h = DenseHeader(ch, None, shape[0], shape[1])

    h.store_matrix(m, f)


def load(path):
    with open(path, 'rb') as f:
        return load_stream(f)


def store(m, path):
    with open(path, 'wb') as f:
        store_stream(m, f)
