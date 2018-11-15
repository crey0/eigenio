#include "catch.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "EigenIO/EigenIO.h"

Eigen::MatrixXd md_dynamic = Eigen::MatrixXd::Random(1000, 100);
Eigen::Matrix<double, 10, 100> md_static = Eigen::Matrix<double, 10, 100>::Random();
Eigen::Matrix<float, 10, 100> mf_static = Eigen::Matrix<float, 10, 100>::Random();
Eigen::Matrix<int32_t, 10, 100> mi32_static = Eigen::Matrix<int32_t, 10, 100>::Random();
Eigen::Matrix<int64_t, 10, 100> mi64_static = Eigen::Matrix<int64_t, 10, 100>::Random();



TEST_CASE("IO Errors")
{
  Eigen::MatrixXd mdfail(0,0);
  CHECK_NOTHROW(EigenIO::store("test.mat", mdfail));
  CHECK_NOTHROW(EigenIO::load("test.mat", mdfail));
  CHECK_THROWS_WITH(EigenIO::load("tist.mat", mdfail), Catch::Contains("IO Error"));

  Eigen::SparseMatrix<double> msfail(0,0);
  CHECK_NOTHROW(EigenIO::store("test.mat", msfail));
  CHECK_NOTHROW(EigenIO::load("test.mat", msfail));
  CHECK_THROWS_WITH(EigenIO::load("tist.mat", msfail), Catch::Contains("IO Error"));
}


TEST_CASE("Dense matrix storage")
{
  SECTION("Matrix store")
  {
      CHECK_NOTHROW(EigenIO::store("matrix_double_dynamic_1000_100.mat", md_dynamic));
      CHECK_NOTHROW(EigenIO::store("matrix_double_static_10_100.mat", md_static));
      CHECK_NOTHROW(EigenIO::store("matrix_float_static_10_100.mat", mf_static));
      CHECK_NOTHROW(EigenIO::store("matrix_int32_static_10_100.mat", mi32_static));
      CHECK_NOTHROW(EigenIO::store("matrix_int64_static_10_100.mat", mi64_static));
  }

  SECTION("Matrix load: scalar type (double)")
  {
      Eigen::MatrixXd md_dynamic_load;
      CHECK_NOTHROW(EigenIO::load("matrix_double_dynamic_1000_100.mat", md_dynamic_load));

      REQUIRE_THAT((md_dynamic-md_dynamic_load).norm(), Catch::WithinAbs(0.0,1e-5));
      
      REQUIRE(md_dynamic.isApprox(md_dynamic_load));

      Eigen::MatrixXf mf_dymamic_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_double_dynamic_1000_100.mat",
					      mf_dymamic_load),
			Catch::Contains("expected scalar size"));

      Eigen::MatrixXi mi_dynamic_load;      
      CHECK_THROWS_WITH(EigenIO::load("matrix_double_dynamic_1000_100.mat",
					      mi_dynamic_load),
			Catch::Contains("wrong scalar type"));

  }

  SECTION("Matrix load: scalar type (float)")
  {
      Eigen::MatrixXf mf_dynamic_load;
      CHECK_NOTHROW(EigenIO::load("matrix_float_static_10_100.mat", mf_dynamic_load));

      REQUIRE(mf_static.isApprox(mf_dynamic_load));

      Eigen::MatrixXd md_dymamic_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_float_static_10_100.mat",
					      md_dymamic_load),
			Catch::Contains("expected scalar size"));

      Eigen::MatrixXi mi_dynamic_load;      
      CHECK_THROWS_WITH(EigenIO::load("matrix_float_static_10_100.mat",
					      mi_dynamic_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: scalar type (int 32)")
  {
      Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> mi32_dynamic_load;
      CHECK_NOTHROW(EigenIO::load("matrix_int32_static_10_100.mat", mi32_dynamic_load));

      REQUIRE(mi32_static.isApprox(mi32_dynamic_load));

      Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> mi64_dymamic_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_int32_static_10_100.mat",
					      mi64_dymamic_load),
			Catch::Contains("expected scalar size"));

      Eigen::MatrixXd md_dymamic_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_int32_static_10_100.mat",
					      md_dymamic_load),
			Catch::Contains("wrong scalar type"));

      Eigen::MatrixXf mf_dynamic_load;      
      CHECK_THROWS_WITH(EigenIO::load("matrix_int32_static_10_100.mat",
					      mf_dynamic_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: scalar type (int 64)")
  {
      Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> mi64_dynamic_load;
      CHECK_NOTHROW(EigenIO::load("matrix_int64_static_10_100.mat", mi64_dynamic_load));

      REQUIRE(mi64_static.isApprox(mi64_dynamic_load));

      Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> mi32_dymamic_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_int64_static_10_100.mat",
					      mi32_dymamic_load),
			Catch::Contains("expected scalar size"));

      Eigen::MatrixXd md_dymamic_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_int64_static_10_100.mat",
					      md_dymamic_load),
			Catch::Contains("wrong scalar type"));

      Eigen::MatrixXf mf_dynamic_load;      
      CHECK_THROWS_WITH(EigenIO::load("matrix_int64_static_10_100.mat",
					      mf_dynamic_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: static size")
  {
      Eigen::Matrix<double, 10, 100> md_static_load;
      CHECK_NOTHROW(EigenIO::load("matrix_double_static_10_100.mat", md_static_load));

      REQUIRE(md_static.isApprox(md_static_load));

	    
      Eigen::Matrix<double, 10, 101> md_static_wrong_cols;
      CHECK_THROWS_WITH(EigenIO::load("matrix_double_static_10_100.mat",
					      md_static_wrong_cols),
			Catch::Contains("wrong number of columns"));

       
      
      Eigen::Matrix<double, 9, 100> md_static_wrong_rows;
      CHECK_THROWS_WITH(EigenIO::load("matrix_double_static_10_100.mat",
					      md_static_wrong_rows),
			Catch::Contains("wrong number of rows"));
  }

  SECTION("Matrix load: wrong matrix type")
  {
      Eigen::SparseMatrix<double> smd_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_double_dynamic_1000_100.mat", smd_load),
			Catch::Contains("wrong matrix type"));
      
  }
}


bool _init_sparse = false;
Eigen::SparseMatrix<double> smd;
Eigen::SparseMatrix<float> smf;
Eigen::SparseMatrix<int32_t> smi32;
Eigen::SparseMatrix<int64_t> smi64;

template<typename Scalar, typename T>
void fill_sparse_matrix(T& m, float fill_in)
{
  typedef Eigen::Triplet<Scalar> Triplet;
  std::vector<Triplet> tripletList;
  auto rows = m.rows();
  auto cols = m.cols();
  tripletList.reserve(fill_in*1.2*rows*cols);
  Eigen::MatrixXf rtake = (1.0+Eigen::MatrixXf::Random(rows, cols).array()).matrix()/2.0;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> rfill = \
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(rows, cols);
  for(Eigen::Index r=0; r<rows; r++)
  {
      for(Eigen::Index c=0; c<cols; c++)
      {
	  if(rtake(r,c) < fill_in)
	    tripletList.push_back(Triplet(r,c, rfill(r,c)));
      }
  }
  m.setFromTriplets(tripletList.begin(), tripletList.end());
}

void init_sparse()
{
  if(_init_sparse) return;

  smd = Eigen::SparseMatrix<double>(100,1000);
  fill_sparse_matrix<double>(smd,0.3);
  smf = Eigen::SparseMatrix<float>(100,1000);
  fill_sparse_matrix<float>(smf,0.3);
  smi32 = Eigen::SparseMatrix<int32_t>(100,1000);
  fill_sparse_matrix<int32_t>(smi32,0.3);
  smi64 = Eigen::SparseMatrix<int64_t>(100,1000);
  fill_sparse_matrix<int64_t>(smi64,0.3);

  
  _init_sparse = true;
}
TEST_CASE("Sparse Matrix Storage")
{
  init_sparse();

  SECTION("Matrix Store")
  {
      CHECK_NOTHROW(EigenIO::store("matrix_sparse_double.mat", smd));
      CHECK_NOTHROW(EigenIO::store("matrix_sparse_float.mat", smf));
      CHECK_NOTHROW(EigenIO::store("matrix_sparse_int32.mat", smi32));
      CHECK_NOTHROW(EigenIO::store("matrix_sparse_int64.mat", smi64));
  }

  SECTION("Matrix load: scalar type (double)")
  {
      Eigen::SparseMatrix<double> smd_load;
      CHECK_NOTHROW(EigenIO::load("matrix_sparse_double.mat",smd_load));

      REQUIRE(smd.isApprox(smd_load));

      Eigen::SparseMatrix<float> smf_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_double.mat",
					      smf_load),
			Catch::Contains("expected scalar size"));

      Eigen::SparseMatrix<int32_t> smi32_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_double.mat",
					      smi32_load),
			Catch::Contains("wrong scalar type"));

      Eigen::SparseMatrix<int64_t> smi64_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_double.mat",
					      smi64_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: scalar type (float)")
  {
      Eigen::SparseMatrix<float> smf_load;
      CHECK_NOTHROW(EigenIO::load("matrix_sparse_float.mat",smf_load));

      REQUIRE(smf.isApprox(smf_load));

      Eigen::SparseMatrix<double> smd_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_float.mat",
					      smd_load),
			Catch::Contains("expected scalar size"));

      Eigen::SparseMatrix<int32_t> smi32_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_float.mat",
					      smi32_load),
			Catch::Contains("wrong scalar type"));

      Eigen::SparseMatrix<int64_t> smi64_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_float.mat",
					      smi64_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: scalar type (int 32)")
  {
      Eigen::SparseMatrix<int32_t> smi32_load;
      CHECK_NOTHROW(EigenIO::load("matrix_sparse_int32.mat",smi32_load));

      REQUIRE(smi32.isApprox(smi32_load));

      Eigen::SparseMatrix<int64_t> smi64_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_int32.mat",
					      smi64_load),
			Catch::Contains("expected scalar size"));

      Eigen::SparseMatrix<float> smf_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_int32.mat",
					      smf_load),
			Catch::Contains("wrong scalar type"));

      Eigen::SparseMatrix<double> smd_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_int32.mat",
					      smd_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: scalar type (int 64)")
  {
      Eigen::SparseMatrix<int64_t> smi64_load;
      CHECK_NOTHROW(EigenIO::load("matrix_sparse_int64.mat",smi64_load));

      REQUIRE(smi64.isApprox(smi64_load));

      Eigen::SparseMatrix<int32_t> smi32_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_int64.mat",
					      smi32_load),
			Catch::Contains("expected scalar size"));

      Eigen::SparseMatrix<float> smf_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_int64.mat",
					      smf_load),
			Catch::Contains("wrong scalar type"));

      Eigen::SparseMatrix<double> smd_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_int64.mat",
					      smd_load),
			Catch::Contains("wrong scalar type"));
  }

  SECTION("Matrix load: wrong matrix type")
  {
      Eigen::MatrixXd md_load;
      CHECK_THROWS_WITH(EigenIO::load("matrix_sparse_double.mat", md_load),
			Catch::Contains("wrong matrix type"));
      
  }
}
