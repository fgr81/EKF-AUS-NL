#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "EKF_AUS_NL.C"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(ekfaus,m) {
	m.doc() = "pybind11 perfect-test";
	py::function NonLinH;
	py::class_<EKF_AUS>(m, "EkfAus")
		.def(py::init<long,long,long,long>())
		.def("LFactor", &EKF_AUS::Lfactor, py::arg("in"))
		.def("TotNumberPert", &EKF_AUS::TotNumberPert)
		.def("HalfNumberNonLinPert", &EKF_AUS::HalfNumberNonLinPert)
		.def("AddInflation", &EKF_AUS::AddInflation, py::arg("in"))
		
		.def("SetModelErrorVariable", &EKF_AUS::SetModelErrorVariable, "istart"_a, "iend"_a, "error"_a.noconvert(), "Xa"_a.noconvert())
    		//.def("PrepareForEvolution",(Eigen::Ref<Eigen::MatrixXd> (EKF_AUS::*)(Eigen::Ref<Eigen::MatrixXd>&, Eigen::Ref<Eigen::MatrixXd>&, Eigen::Ref<Eigen::MatrixXd>&)) &EKF_AUS::PrepareForEvolution, "analysis"_a.noconvert(), "Xa"_a.noconvert(), "gmunu"_a.noconvert())
		.def("PrepareForEvolution", &EKF_AUS::PrepareForEvolution, "analysis"_a.noconvert(), "Xa"_a, "gmunu"_a.noconvert(),py::keep_alive<1, 3>())
		.def("PrepareForAnalysis", &EKF_AUS::PrepareForAnalysis, "xf"_a.noconvert(), "Evoluted"_a.noconvert(), "Gmunu"_a.noconvert())

		.def("Assimilate", &EKF_AUS::Assimilate2, "measure"_a.noconvert(),"NonLinH"_a, "R"_a.noconvert(), "xf"_a.noconvert(), "Xf"_a.noconvert())
		// .def("Assimilate", &EKF_AUS::Assimilate2,"measure"_a.noconvert(), "NonLinH"_a.noconvert(),  "R"_a.noconvert(), "xf"_a.noconvert(), "Xf"_a.noconvert())
		
		// m.def("func_arg", &func_arg);
		.def("linM", (long (EKF_AUS::*)()) &EKF_AUS::linM)
		.def("N", (long (EKF_AUS::*)()) &EKF_AUS::N)
		.def("N", (void (EKF_AUS::*)(long int)) &EKF_AUS::N, py::arg("n"))
		.def("P", (long (EKF_AUS::*)()) &EKF_AUS::P)
		.def("P", (void (EKF_AUS::*)(long int)) &EKF_AUS::P, py::arg("new_p"))		
	;

}
