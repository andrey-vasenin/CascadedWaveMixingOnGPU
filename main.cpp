#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include "solver_v2.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

using namespace std;
namespace py = pybind11;

auto make_tlist(float &time_delta, float domega, int LEN=10000)
{
	float T = 2.f * M_PI / domega;
	time_delta = T / static_cast<float>(LEN);
	vector<float> tlist(LEN);
	generate(tlist.begin(), tlist.end(),
		[k = 0, &time_delta] () mutable { return time_delta * k++; });
	return tlist;
}

vector<complex<float>> calc(float domega, float omega1, float omega2, float gamma1,
	float eta1, float gamma2, float alpha, vector<float> Wvals, vector<float> Evals, vector<int> orders,
	int LEN)
{
	using namespace std::complex_literals;
	ModelParams params = {
		.domega = domega,
		.omega1 = omega1,
		.omega2 = omega2,
		.gamma1 = gamma1,
		.eta1 = eta1,
		.gamma2 = gamma2,
		.alpha = alpha
	};
	auto time_delta = .001f;
	auto tlist = make_tlist(time_delta, params.domega, LEN);
	Solver s(params, time_delta, tlist.size(), Wvals, Evals, orders);
	s.compute(tlist, params);
	return s.get_fft_values();
}

PYBIND11_MODULE(solver, m) {
    m.def("calc", &calc);
}