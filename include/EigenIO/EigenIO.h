
/** fileio.h
 *  @author: Christophe Reymann
 *  
 *  This Source Code Form is subject to the terms of the Mozilla Public License,
 *  v. 2.0. If a copy of the MPL was not distributed with this file, You can
 *  obtain one at http://mozilla.org/MPL/2.0/.
 * 
 *
 *  Store and load Eigen matrices in binary format, supports both dense and
 *  sparse matrices.
 *
 *  The binary format layout is as follows:
 *    |common header|additional header|data| 
 * 
 *  Common header: 
 *    Layout:        |format|scalar size|
 *    Size in bytes: |  1   |     1     |
 *
 *    Format: This byte describes the matrix type
 *      Bit 0: 0 --> Dense matrix
 *             1 --> Sparse matrix
 * 
 *      Bit 1: 0 --> Integral scalar type
 *             1 --> Floating point scalar type
 *
 *      Bits 2-7: currently unused.
 *
 *    Scalar type: the size of the scalar type in bytes. The maximum scalar size
 *    is thefore 255 bytes.
 *
 *
 *  Dense matrices
 *
 *  Additional header:
 *    Layout:        |rows|cols|
 *    Size in bytes: |  4 |  4 |
 * 
 *    Rows: the number of rows, up to 2^32-1
 *    Cols: the number of columns, upt to 2^32-1
 * 
 *    Data: the contiguous data array, of byte size rows * cols * scalar size.
 *
 * 
 *  Sparse Matrices
 *
 *  Sparse matrices are currently stored in CCS format only.
 *  
 *  Additional header:
 *    Layout:        |SIS|SNNZ|sinner|souter|
 *    Size in bytes: | 1 |  4 |   4  |   4  |
 *
 *    SIS: Storage Index Size in bytes
 *    SNNZ: the number of non zero entries
 *    sinner: the inner size, up to 2^32-1 (number of rows in CCS)
 *    souter: the outer size, up to 2^32-1 (number of cols in CCS)
 * 
 *  Data:
 *    Layout: |NNZ|inner|outer|
 *
 *    NNZ: the non zero entries array of size SNNZ * salar size bytes
 *    inner: the inner array, of size SNNZ * SIS bytes
 *    outer: the outer array, of size souter * SIS bytes
 */
#ifndef EIGENIO_H
#define EIGENIO_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace EigenIO {


template<typename T> struct is_sparse_matrix
{
  static constexpr bool value = std::is_base_of<typename Eigen::SparseMatrixBase<T>, T>::value;
};

template<typename T> struct is_dense_matrix
{
  static constexpr bool value = std::is_base_of<typename Eigen::PlainObjectBase<T>, T>::value;
};

enum  MatrixFormat : int {
			MATRIX_TYPE=0,
			SCALAR_TYPE=1,
};
enum  MatrixType : bool {
			DENSE_MATRIX=false,
			SPARSE_MATRIX=true,
};
  
enum  ScalarType : bool {
		      INTEGRAL=false,
		      FLOATING_POINT=true
};

void set_bit(char & val, int num, bool bitval)
{
    val = (val & ~(1<<num)) | (bitval << num);
}

bool get_bit(char  val, int num)
{
  return val & (1 << num);
}

void check_stream(const std::basic_ios<char> &stream)
{
  if(!stream)
  {
      throw std::runtime_error("IO Error: stream failed.");
  }
}

template<typename Scalar>
void check_scalar_type(char format)
{
  char scalar_type = get_bit(format, SCALAR_TYPE);  
  if((scalar_type != FLOATING_POINT && scalar_type != INTEGRAL)
     || (scalar_type == FLOATING_POINT && std::is_integral<Scalar>::value)
     || (scalar_type == INTEGRAL && std::is_floating_point<Scalar>::value))
    throw std::runtime_error("Matrix load failed: wrong scalar type.");
}

template<typename Scalar>
void check_scalar_size(char scalar_size)
{
  if(scalar_size != sizeof(Scalar))
    throw std::runtime_error("Matrix load failed: expected scalar size "
			     + std::to_string(sizeof(Scalar))
			     + " got "
			     + std::to_string(scalar_size) +".");
}

template<typename StorageIndex>
void check_storage_index_size(char storage_index_size)
{
  if(storage_index_size != sizeof(StorageIndex))
    throw std::runtime_error("Matrix load failed: expected StorageIndex size "
			     + std::to_string(sizeof(StorageIndex))
			     + " got "
			     + std::to_string(storage_index_size) +".");
}


template<typename T> void read_matrix_header(std::ifstream &stream, T& m)
{
  if(m.IsRowMajor) throw std::invalid_argument("Matrix store failed: row major not supported."); 

  char format = stream.get();
  
  if(get_bit(format, MATRIX_TYPE) != is_sparse_matrix<T>::value)
    throw std::runtime_error("Matrix load failed: wrong matrix type.");

  check_scalar_type<typename T::Scalar>(format);

  char scalar_size = stream.get();
  check_scalar_size<typename T::Scalar>(scalar_size); 
}

template<typename T> void read_matrix_dense_header(std::ifstream &stream, T& m)
{ 
  uint32_t rows, cols;
  stream.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));  
  stream.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));

  if(m.cols() != cols || m.rows() != rows)
  {
      if(m.cols() != cols && m.ColsAtCompileTime != Eigen::Dynamic)
	throw std::length_error("Matrix load failed: wrong number of columns, and size not dynamic.");

      if(m.rows() != rows && m.RowsAtCompileTime != Eigen::Dynamic)
	throw std::length_error("Matrix load failed: wrong number of rows, and size not dynamic.");

      m.resize(rows, cols);
  }
  
}

template<typename T> void read_matrix_sparse_header(std::ifstream &stream, T& m)
{
  char storage_index_size = stream.get();
  check_storage_index_size<typename T::StorageIndex>(storage_index_size);
  
  int32_t s_nnz, s_inner, s_outer;
  stream.read(reinterpret_cast<char *>(&s_nnz), sizeof(int32_t));
  stream.read(reinterpret_cast<char *>(&s_inner), sizeof(int32_t));
  stream.read(reinterpret_cast<char *>(&s_outer), sizeof(int32_t));

  //Only CCS supported, for CSR support rows and columns size should be swapped
  m.conservativeResize(s_inner, s_outer);
  m.makeCompressed();
  m.data().resize(s_nnz);
}

template<typename T> void write_matrix_header(std::ofstream &stream, const T& m)
{
  if(m.IsRowMajor) throw std::invalid_argument("Matrix store failed: row major not supported."); 

  char format = 0;
  set_bit(format, MATRIX_TYPE, is_sparse_matrix<T>::value);
  set_bit(format, SCALAR_TYPE,
	  std::is_floating_point<typename T::Scalar>::value ? FLOATING_POINT : INTEGRAL);
  stream.put(format);
  
  stream.put(sizeof(typename T::Scalar));
}

template<typename T> void write_matrix_dense_header(std::ofstream &stream, const T& m)
{
  uint32_t rows = m.rows();
  uint32_t cols = m.cols();
  stream.write(reinterpret_cast<const char *>(&rows), sizeof(uint32_t));
  stream.write(reinterpret_cast<const char *>(&cols), sizeof(uint32_t));
}

template<typename T> void write_matrix_sparse_header(std::ofstream &stream, const T& m)
{  
  int32_t s_nnz, s_inner, s_outer;
  s_nnz = m.nonZeros();
  s_inner = m.innerSize(); // number of rows in CCS
  s_outer = m.outerSize(); // number of columns in CCS

  stream.put(sizeof(typename T::StorageIndex));
  stream.write(reinterpret_cast<const char *>(&s_nnz), sizeof(uint32_t));
  stream.write(reinterpret_cast<const char *>(&s_inner), sizeof(uint32_t));
  stream.write(reinterpret_cast<const char *>(&s_outer), sizeof(uint32_t));
  
}

/**
 * @name store - Stores an Eigen Dense matrix (see store(std::string, T))
 *  WARNING: The stream should be binary, if not it will result
 *  in possibly undetectable file corruption.
 * @param stream -  A binary output file stream, open 
 * @param m -  The matrix
 * @return void
 */
template<typename T>
std::enable_if_t<is_dense_matrix<T>::value>
store(std::ofstream &stream, const T &m)
{
  check_stream(stream);

  write_matrix_header(stream, m);
  write_matrix_dense_header(stream, m);
  stream.write(reinterpret_cast<const char *>(m.data()), sizeof(typename T::Scalar) * m.size());
  stream.flush();
}

/**
 * @name load - Loads an Eigen Dense matrix (see store(std::string, T))
 *  WARNING: The stream should be binary, if not it will result
 *  in possibly undetectable file corruption.
 * @param stream -  A binary output file stream, open 
 * @param m -  The matrix
 * @return void
 */
template<typename T>
std::enable_if_t<is_dense_matrix<T>::value>
load(std::ifstream &stream, T &m)
{
  check_stream(stream);

  read_matrix_header(stream, m);
  read_matrix_dense_header(stream, m);
  stream.read(reinterpret_cast<char *>(m.data()), sizeof(typename T::Scalar) * m.size());
}

/**
 * @name store - Stores an Eigen Sparse matrix (see store(std::string, T))
 *  WARNING: The stream should be binary, if not it will result
 *  in possibly undetectable file corruption.
 * @param stream -  A binary output file stream, open 
 * @param m -  The matrix
 * @return void
 */
template<typename T>
 std::enable_if_t<is_sparse_matrix<T>::value>
store(std::ofstream &stream, const T &m)
{
  if(!m.isCompressed())
    throw std::runtime_error("Storage of uncompressed sparse matrix is unsupported.");

  check_stream(stream);

  write_matrix_header(stream, m);
  write_matrix_sparse_header(stream, m);
  stream.write(reinterpret_cast<const char *>(m.valuePtr()),
	       sizeof(typename T::Scalar) * m.nonZeros());
  stream.write(reinterpret_cast<const char *>(m.innerIndexPtr()),
	       sizeof(typename T::StorageIndex) * m.nonZeros());
  stream.write(reinterpret_cast<const char *>(m.outerIndexPtr()),
	       sizeof(typename T::StorageIndex) * (m.outerSize() + 1));
  
  stream.flush();
}

/**
 * @name load - Loads an Eigen sparse matrix (see load(std::string, T))
 *  WARNING: The stream should be binary, if not it will result
 *  in possibly undetectable file corruption.
 * @param stream -  A binary input file stream, open 
 * @param m -  The matrix
 * @return void
 */
template<typename T>
std::enable_if_t<is_sparse_matrix<T>::value>
load(std::ifstream &stream, T &m)
{
  check_stream(stream);
  
  read_matrix_header(stream, m);
  read_matrix_sparse_header(stream, m);
  auto nnz = m.data().allocatedSize();
  stream.read(reinterpret_cast<char *>(m.valuePtr()), sizeof(typename T::Scalar) * nnz);
  stream.read(reinterpret_cast<char *>(m.innerIndexPtr()),
	      sizeof(typename T::StorageIndex) * nnz);
  stream.read(reinterpret_cast<char *>(m.outerIndexPtr()),
	      sizeof(typename T::StorageIndex) * (m.outerSize() + 1));
}



/**
 * @name store - Stores an Eigen matrix to a binary file  Both dense and sparse
 *  are supported, but only  column major/CCS. 
 *  Dense matrices should be compressed beforehand.
 * @param file -  The path of the file
 * @param obj -  The matrix.
 * @return void
 */
template<class T> void store(const std::string &file, const T& obj)
{
  auto stream = std::ofstream(file, std::ofstream::binary);
  store(stream, obj);
}
/**
 * @name load - Loads an Eigen matrix stored in binary format.
 *  Throws an exception on errors, in particular if the Scalar type
 *  is wrong, or if the matrix is fixed size of the wrong size or if
 *  trying to load the wrong type of matrix (dense in sparse or sparse 
 *   in dense).
 * @param file - The path of the file
 * @param obj -  The matrix in which the file will be loaded.
 * @return void
 */
template<class T> void load(const std::string &file, T& obj)
{
  auto stream = std::ifstream(file, std::ifstream::binary);
  load(stream, obj);
}

}  // end namespaces

#endif // EIGENIO_H
