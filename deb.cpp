#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include "solver_v6.h"
#include "timer.h"

using namespace std;

int LEN = 10000;

auto make_tlist(float &time_delta, float domega)
{
	float T = 4.f * M_PI / domega;
	time_delta = T / static_cast<float>(LEN);
	vector<float> tlist(LEN);
	generate(tlist.begin(), tlist.end(), [k = 0, &time_delta] () mutable { return time_delta * k++; });
	return tlist;
}

template <typename T>
void print_vector(string&& name, const vector<T>& v, int N=-1)
{
	cout << "Vector " << name << ":\n";
	// cout.precision(3);
	if (N == -1)
		for (auto iter = v.begin(); iter != v.end(); iter++)
			cout << *iter << " ";
	else
		for (auto iter = v.begin(); iter != v.begin() + N; iter++)
			cout << *iter << " ";
	cout << endl;
}

template <typename T>
void print_matrix(string&& name, const vector<T>& v, int N)
{
	cout << "Matrix " << name << ":\n";
	cout.precision(2);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cout << v[i * N + j] << '\t';
		}
		cout << endl;
	}
	cout << endl;
}

void check_dm(const vector<float>&& dm)
{
	int SIZE = dm.size() / 15;
	vector<float> dm_traces(SIZE);
	for (int i = 0; i < SIZE; i++)
	{
		dm_traces[i] = dm[i * SIZE + 4] + dm[i * SIZE + 9] + dm[i * SIZE + 14];
	}
	print_vector("dm_traces", dm_traces);
}

template <typename T>
void save_vector(const vector<T>& v, string&& filename)
{
	ofstream outfile(filename,
		ofstream::out | ofstream::binary | ofstream::trunc);
	outfile.write(reinterpret_cast<const char*>(v.data()),
		v.size() * sizeof(T));
	// copy(v.begin(), v.end(), ostream_iterator<T>(outfile));
	outfile.close();
	cout << "Vector length: " << v.size() << endl;
	cout << "Vector capacity: " << v.capacity()  << endl;
}

// void read_vector(string&& filename, unsigned int size)
// {
// 	ifstream is(filename, ifstream::in | ifstream::binary);
// 	float* vec = new float[size];
// 	vector<float> numbers(start, end);
// 	cout << "Read " << numbers.size() << " numbers" << endl;
// 	// print the numbers to stdout
// 	cout << "numbers read in:\n";
// 	// std::copy(numbers.begin(), numbers.end(),
// 	// 			std::ostream_iterator<float>(std::cout, " "));
// 	print_vector("read vector", numbers, 15);
// 	delete [] vec;
// 	is.close();
// }

vector<complex<float>> calc(const float domega, const float omega1, const float omega2, const float gamma1,
	const float eta1, const float gamma2, const float alpha, const vector<float> &Wvals,
	const vector<float> &Evals, const vector<int> orders)
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
	auto tlist = make_tlist(time_delta, params.domega);
	cout << "Time delta: " << time_delta << " us\n";
	Timer tinit("init");
	Timer tcompute("compute");
	Timer tfft("get_fft_values");
	tinit.resume();
	int CUT = LEN;
	vector<float> tlist2(CUT);
	for (int i = 0; i < CUT; i++)
		tlist2[i] = tlist[i];
	Solver s(params, time_delta, tlist2.size(), Wvals, Evals, orders);
	tinit.pause();
	// print_vector("dm0", s.get_dm(), 16);
	// print_vector("dm0_batched", s.get_dm_batched(), 16 * 3);
	tcompute.resume();
	s.compute(tlist2, params);
	// auto dms = s.compute_with_dmsave(tlist2, params);
	// save_vector(dms, "dms.dt");
	// dms.clear();
	// read_vector("dms.dt");
	// print_vector("dm00", dms, 15);
	// s.setM();
	// s.setMd1(params, 0);
	// s.setMd2(params, 0);
	// print_vector("dm", s.get_dm_batched(), 15);
	// check_dm(s.get_dm_batched());
	// s.compute_next_step();
	// print_matrix("dm", s.get_dm_batched(), 16);
	// s.add_to_fourier(0);
	// print_vector("sines", s.get_sines());
	// print_vector("cosines", s.get_cosines());
	// print_matrix("dm_fft_cos", s.get_dm_fft_cos(), 16);
	// print_matrix("dm_fft_sin", s.get_dm_fft_sin(), 16);

	// print_vector("dot_re", s.get_dot_re());
	// print_vector("dot_im", s.get_dot_im());
	// print_matrix("fft_re", s.get_fft_re(), 4);
	// print_matrix("fft_im", s.get_fft_im(), 4);
	// s.compute_sigma_minus();
	// print_matrix("fft_re", s.get_fft_re(), 4);
	// print_matrix("fft_im", s.get_fft_im(), 4);
	// print_vector("temp_dm", s.get_temp_dm(), 45);
	// print_matrix("M", s.get_M(), 16);
	// print_matrix("M0", s.get_M0(), 15 * 2);
	// print_matrix("M0", s.get_M0_host(), 16);
	// print_matrix("Md1", s.get_Md1(), 30);
	// print_matrix("Md2", s.get_Md2(), 30);
	tcompute.pause();
	tfft.resume();
	// print_vector("dm", s.get_dm_batched(), 16);
	auto fft = s.get_fft_values();
	// print_matrix("fft", fft, 4);
	tfft.pause();
	tinit.result();
	tcompute.result();
	tfft.result();
	// cout << fft[0] << endl;
	// save_vector(fft, "fft.dt");
	return fft;
}

int main()
{
	Timer t0("main");
	t0.resume();
	float domega = 2 * M_PI * 0.2;
	float omega1 = 0;
	float omega2 = 0;
	float gamma1 = 2 * M_PI * 3;
	float eta1 = 2 * M_PI * 0.1;
	float gamma2 = 2 * M_PI * 3;
	float alpha = 0.5;
	vector<float> Wvals(100);
	vector<float> Evals(100);
	for (int i = 0; i < Wvals.size(); i++)
	{
		Wvals[i] = pow(10.f, -2.f + static_cast<float>(i) * .04f);
		Evals[i] = Wvals[i];
	}
	vector<int> orders = {-5, -3, 3, 5};
	auto res = calc(domega, omega1, omega2, gamma1, eta1, gamma2, alpha, Wvals, Evals, orders);
	cout << res[0] << endl;
	t0.pause();
	t0.result();
	return 0;
}