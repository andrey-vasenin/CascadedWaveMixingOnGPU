#ifndef _SOLVER_
#define _SOLVER_
#define __HIP_PLATFORM_HCC__
#define __DEBUG__
#include <vector>
#include <cmath>
#include <iostream>
#include <cblas.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <fftw3.h>
#include <complex>
#include <string>
#include <algorithm>
#include <execution>
#include "timer.h"
using namespace std;

const int N { 16 };
const int totalN { N  *  N };

struct ModelParams
{
	float domega;
	float omega1;
	float omega2;
	float gamma1;
	float eta1;
	float gamma2;
	float alpha;
};

void checkHipError(hipError_t&& stat)
{
#ifdef __DEBUG__
	if (stat != hipSuccess)
	{
		cerr << hipGetErrorName(stat) << ": " << hipGetErrorString(stat) << endl;
	}
#endif
}

void checkRocblasError(rocblas_status&& stat)
{
#ifdef __DEBUG__
	if (stat != rocblas_status_success)
	{
		cerr << rocblas_status_to_string(stat) << endl;
	}
#endif
}

void log(string&& str)
{
#ifdef __LDEBUG__
	cout << str << endl;
#endif
}

class Solver
{
// Host variables
	vector<float> M0_host;
	vector<float> Md1_host;
	vector<float> Md2_host;
	vector<float> M_host;
	vector<float> dm_host;
	float delta_t;
	float delta_n;
	float minus_delta_n;
	const float one = 1;
	const float minusone = -1;
	const float zero = 0;
	int LEN;
	int ESIZE;
	int WSIZE;
	int SIZE;
	int ORDERS_NUM = 1;
	vector<float> Es_host;
	vector<float> Ws_host;
	vector<float> ones_host;
	vector<float> frequencies;
	vector<float> cosines_host;
	vector<float> sines_host;
// GPU variables
	float* M0;
	float* M0_batched;
	float* Md1;
	float* Md2;
	float* M_batched;
	float* dm;
	float* dm_batched;
	float* dot_re;
	float* dot_im;
	float* sigmas_re;
	float* sigmas_im;
	float* Es;
	float* Ws;
	float* ones;
	float* cosines;
	float* sines;
	float* fft_re;
	float* fft_im;
	rocblas_handle handle;
// Debug variables
	Timer tsetM;
	Timer tsetMd1;
	Timer tsetMd2;
	Timer tnextStep;
	Timer taddtofft;
	Timer tcompsigma;
	Timer tsetsines;
	Timer tsger;
	Timer tforsin;
	Timer tsinmemcpy;

public:
	Solver(const ModelParams &pars, float dt, int length, const vector<float> &Wvals, const vector<float> &Evals, const vector<int> &orders) :
		M0_host(totalN, 0.f), Md1_host(totalN, 0.f), Md2_host(totalN, 0.f),
		M_host(totalN, 0.f), dm_host(N, 0.f),
		Es_host(Wvals.size() * Evals.size(), 0.f), Ws_host(Wvals.size() * Evals.size(), 0.f),
		ones_host(Wvals.size() * Evals.size(), 1.f), frequencies(orders.size(), 0.f),
		cosines_host(orders.size(), 0.f), sines_host(orders.size(), 0.f),
		tsetM("setM"), tsetMd1("setMd1"), tsetMd2("setMd2"), tnextStep("compute_next_step"),
		taddtofft("add_to_fourier"), tcompsigma("compute_sigma_minus"),
		tsetsines("set_sines_cosines"), tsger("rocblas_sger"),
		tforsin("for loop with sin"), tsinmemcpy("memcpy sin")
	{
    	checkRocblasError(rocblas_create_handle(&handle));
		dm_host[0] = 1.f;
		delta_t = dt;
		LEN = length;
		delta_n = 1.f / static_cast<float>(LEN);
		minus_delta_n = -delta_n;
		ESIZE = Evals.size();
		WSIZE = Wvals.size();
		for (int i = 0; i < WSIZE; i++)
		{
			for (int j = 0; j < ESIZE; j++)
			{
				Es_host[i * ESIZE + j] = Evals[j];
				Ws_host[i * ESIZE + j] = Wvals[i];
			}
		}
		SIZE = ESIZE * WSIZE;
		ORDERS_NUM = orders.size();
		log("Mallocs");
		checkHipError(hipMalloc(&M0, totalN * sizeof(float)));
		checkHipError(hipMalloc(&M0_batched, SIZE * totalN * sizeof(float)));
		checkHipError(hipMalloc(&Md1, totalN * sizeof(float)));
		checkHipError(hipMalloc(&Md2, totalN * sizeof(float)));
		checkHipError(hipMalloc(&M_batched, SIZE * totalN * sizeof(float)));
		checkHipError(hipMalloc(&dm, N * sizeof(float)));
		checkHipError(hipMalloc(&dm_batched, SIZE * N * sizeof(float)));
		checkHipError(hipMalloc(&dot_re, N * sizeof(float)));
		checkHipError(hipMalloc(&dot_im, N * sizeof(float)));
		checkHipError(hipMalloc(&sigmas_re, SIZE * sizeof(float)));
		checkHipError(hipMalloc(&sigmas_im, SIZE * sizeof(float)));
		checkHipError(hipMalloc(&Es, SIZE * sizeof(float)));
		checkHipError(hipMalloc(&Ws, SIZE * sizeof(float)));
		checkHipError(hipMalloc(&ones, SIZE * sizeof(float)));
		checkHipError(hipMalloc(&cosines, ORDERS_NUM * sizeof(float)));
		checkHipError(hipMalloc(&sines, ORDERS_NUM * sizeof(float)));
		checkHipError(hipMalloc(&fft_re, SIZE * ORDERS_NUM * sizeof(float)));
		checkHipError(hipMalloc(&fft_im, SIZE * ORDERS_NUM * sizeof(float)));
		log("Memcopy");
		checkHipError(hipMemcpyHtoD(Es, Es_host.data(), SIZE * sizeof(float)));
		checkHipError(hipMemcpyHtoD(Ws, Ws_host.data(), SIZE * sizeof(float)));
		checkHipError(hipMemcpyHtoD(ones, ones_host.data(), SIZE * sizeof(float)));
		log("FFT arrays set");
		checkHipError(hipMemset(fft_re, 0, SIZE * ORDERS_NUM * sizeof(float)));
		checkHipError(hipMemset(fft_im, 0, SIZE * ORDERS_NUM * sizeof(float)));
		checkRocblasError(rocblas_set_vector(N, sizeof(float), dm_host.data(), 1, dm, 1));
		checkRocblasError(rocblas_sger(handle, N, SIZE, &one, dm, 1, ones, 1, dm_batched, N));
		setM0(pars);
    	vector<float> dot_re_host(N, 0.f);
    	dot_re_host[1] = 1;
    	dot_re_host[11] = 1;
    	vector<float> dot_im_host(N, 0.f);
    	dot_im_host[4] = -1;
    	dot_im_host[14] = -1;
    	log("set dot_re and dot_im");
		checkRocblasError(rocblas_set_vector(N, sizeof(float), dot_re_host.data(), 1, dot_re, 1));
		checkRocblasError(rocblas_set_vector(N, sizeof(float), dot_im_host.data(), 1, dot_im, 1));
		setFrequencies(pars.domega, orders);
	}
	~Solver()
	{
		tsetM.result();
		tsetMd1.result();
		tsetMd2.result();
		tnextStep.result();
		tcompsigma.result();
		taddtofft.result();
		tsetsines.result();
		tsger.result();
		tforsin.result();
		tsinmemcpy.result();
		checkRocblasError(rocblas_destroy_handle(handle));
		checkHipError(hipFree(M0));
		checkHipError(hipFree(M0_batched));
		checkHipError(hipFree(Md1));
		checkHipError(hipFree(Md2));
		checkHipError(hipFree(M_batched));
		checkHipError(hipFree(dm));
		checkHipError(hipFree(dm_batched));
		checkHipError(hipFree(dot_re));
		checkHipError(hipFree(dot_im));
		checkHipError(hipFree(sigmas_re));
		checkHipError(hipFree(sigmas_im));
		checkHipError(hipFree(Es));
		checkHipError(hipFree(Ws));
		checkHipError(hipFree(ones));
		checkHipError(hipFree(cosines));
		checkHipError(hipFree(sines));
		checkHipError(hipFree(fft_re));
		checkHipError(hipFree(fft_im));
	}

	void setM0(const ModelParams &pars)
	{
		log("set M0");
		M0_host = {-pars.gamma1 - pars.gamma2 - pars.eta1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			0.f, -pars.gamma1 - pars.gamma2/2.f - pars.eta1, 0.f, 0.f, -pars.omega2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, (-pars.alpha) *  sqrt(pars.gamma1 * pars.gamma2), -(pars.gamma1/2.f) - pars.gamma2 - pars.eta1/ 2, 
			0.f, 0.f, 0.f, 0.f, 0.f, -pars.omega1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -(pars.gamma1/2.f) - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.omega1 - pars.omega2, 0.f, 0.f, 0.f, 0.f, pars.omega2, 0.f, 0.f, -pars.gamma1 - pars.gamma2/2.f - pars.eta1, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma2, 0.f, 0.f, 0.f, 0.f, -pars.gamma1 - pars.eta1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			pars.alpha * sqrt(pars.gamma1 * pars.gamma2), 0.f, 0.f, 0.f, 0.f, (-pars.alpha) *  sqrt(pars.gamma1 * pars.gamma2), 
			-(pars.gamma1/2.f) - pars.gamma2/2.f - pars.eta1/ 2, 0.f, 0.f, pars.omega2 - pars.omega1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			pars.alpha * sqrt(pars.gamma1 * pars.gamma2), pars.gamma2, 0.f, 0.f, 0.f, 0.f, -(pars.gamma1/2.f) - pars.eta1/2.f, 0.f, 0.f,
			0.f, 0.f, 0.f, -pars.omega1, 0.f, 0.f, 0.f, 0.f, pars.omega1, 0.f, (-pars.alpha) * sqrt(pars.gamma1 * pars.gamma2), 0.f, 0.f, 0.f,
			-(pars.gamma1/2.f) - pars.gamma2 - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.omega1 - pars.omega2,
			0.f, 0.f, -(pars.gamma1/2.f) - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1, 0.f, 0.f, 0.f, 0.f,
			0.f, -2.f * pars.alpha * sqrt(pars.gamma1 * pars.gamma2), 0.f, 0.f, 0.f, -pars.gamma2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1,
			pars.alpha * sqrt(pars.gamma1 * pars.gamma2), 0.f, 0.f, 0.f, 0.f, (-pars.alpha) * sqrt(pars.gamma1 * pars.gamma2), 0.f, 0.f, 0.f,
			-(pars.gamma2/2.f), 0.f, 0.f, -pars.omega2, 0.f, 0.f, 0.f, 0.f, pars.omega1 + pars.omega2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
			-(pars.gamma1/2.f) - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.alpha * sqrt(pars.gamma1 * pars.gamma2),
			0.f, 0.f, pars.omega1, pars.gamma2, 0.f, 0.f, 0.f, 0.f, -(pars.gamma1/2.f) - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1,
			0.f, 0.f, 0.f, pars.alpha * sqrt(pars.gamma1 * pars.gamma2), 0.f, 0.f, pars.omega2, 0.f, (-pars.alpha) * sqrt(pars.gamma1 * pars.gamma2),
			-(pars.gamma2/2.f), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1, 2 * pars.alpha * sqrt(pars.gamma1 * pars.gamma2),
			0.f, 0.f, 0.f, pars.gamma2, 0.f, 0.f, 0.f, 0.f, 0};
		checkRocblasError(rocblas_set_matrix(N, N, sizeof(float), M0_host.data(), N, M0, N));
		checkRocblasError(rocblas_sger(handle, totalN, SIZE, &one, M0, 1, ones, 1, M0_batched, totalN));
	}

	void setMd1(const ModelParams &pars, float t)
	{
		tsetMd1.resume();
		log("set Md1");
		float wsqrtcos = sqrt(pars.eta1) * cos(pars.domega * t);
		float wsqrtsin = sqrt(pars.eta1) * sin(pars.domega * t);
		Md1_host = {0.f, 0.f, -2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, -2.f*wsqrtsin, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 
			-wsqrtcos, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 
			-wsqrtsin, 0.f, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			wsqrtcos, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 
			0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, wsqrtsin, 
			0.f, 0.f, 0.f, 0.f, wsqrtsin, 0.f, 0.f, -wsqrtsin, 
			0.f, 0.f, wsqrtcos, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 
			-2.f*wsqrtsin, 0.f, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 
			wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 
			0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtcos, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, wsqrtsin, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			wsqrtsin, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 
			0.f, -wsqrtsin, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 0.f, 
			2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*wsqrtsin, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 
			wsqrtcos, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 
			wsqrtsin, 0.f, 0.f, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 
			wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 
			0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtsin, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 
			-wsqrtsin, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 
			-wsqrtcos, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 0.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 
			2.f*wsqrtsin, 0.f, 0.f};
		checkRocblasError(rocblas_set_matrix(N, N, sizeof(float), Md1_host.data(), N, Md1, N));
		checkRocblasError(rocblas_sger(handle, totalN, SIZE, &one, Md1, 1, Ws, 1, M_batched, totalN));
		tsetMd1.pause();
	}

	void setMd2(const ModelParams &pars, float t)
	{
		tsetMd2.resume();
		log("set Md2");
		float esqrtcos = sqrt(pars.gamma2) * cos(pars.domega * t);
		float esqrtsqrt = sqrt(pars.gamma2) * sqrt(pars.domega * t);
		Md2_host = {0.f, -2.f*esqrtcos, 0.f, 0.f, 2.f*esqrtsqrt, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, esqrtcos, 0.f, 0.f, 0.f, 
			0.f, -esqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			-esqrtcos, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 
			-esqrtsqrt, 0.f, 0.f, esqrtsqrt, 0.f, 0.f, 0.f, 
			0.f, 0.f, esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 
			esqrtsqrt, 0.f, 0.f, 0.f, 0.f, -esqrtsqrt, 0.f, 
			0.f, -esqrtsqrt, 0.f, 0.f, 0.f, 0.f, esqrtsqrt, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*esqrtcos, 0.f, 0.f, 
			-2.f*esqrtsqrt, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 
			-esqrtsqrt, 0.f, 0.f, 0.f, 0.f, esqrtsqrt, 
			0.f, 0.f, 0.f, 0.f, 0.f, esqrtcos, 0.f, 0.f, esqrtcos, 
			0.f, 0.f, esqrtsqrt, 0.f, 0.f, -esqrtsqrt, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, -esqrtsqrt, 0.f, 0.f, esqrtsqrt, 
			0.f, 0.f, -esqrtcos, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 
			0.f, 0.f, 0.f, esqrtsqrt, 0.f, 0.f, 0.f, 0.f, -esqrtsqrt, 
			esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.f*esqrtcos, 0.f, 0.f, 
			2.f*esqrtsqrt, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 
			0.f, 0.f, -esqrtsqrt, 0.f, 0.f, 0.f, 0.f, esqrtsqrt, 
			esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 
			0.f, 0.f, 0.f, esqrtsqrt, 0.f, 0.f, -esqrtsqrt, 
			0.f, 0.f, esqrtcos, 0.f, 0.f, esqrtcos, 0.f, 0.f, 
			0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -esqrtsqrt, 0.f, 0.f, 
			0.f, 0.f, esqrtsqrt, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
			2.f*esqrtcos, 0.f, 0.f, -2.f*esqrtsqrt, 0.f};
		checkRocblasError(rocblas_set_matrix(N, N, sizeof(float), Md2_host.data(), N, Md2, N));
		checkRocblasError(rocblas_sger(handle, totalN, SIZE, &one, Md2, 1, Es, 1, M_batched, totalN));
		tsetMd2.pause();
	}

	void setFrequencies(float df, const vector<int> &orders)
	{
		log("set frequencies");
		for (int i = 0; i < ORDERS_NUM; i++)
			frequencies[i] = df * static_cast<float>(orders[i]);
	}

	void set_sines_cosines(float t)
	{
		log("set sines and cosines");
		tforsin.resume();
		for (int i = 0; i < ORDERS_NUM; i++)
		{
			cosines_host[i] = cos(frequencies[i] * t);
			sines_host[i] = sin(frequencies[i] * t);
		}
		tforsin.pause();
		tsinmemcpy.resume();
		checkRocblasError(rocblas_set_vector(ORDERS_NUM, sizeof(float), cosines_host.data(), 1, cosines, 1));
		checkRocblasError(rocblas_set_vector(ORDERS_NUM, sizeof(float), sines_host.data(), 1, sines, 1));
		// checkHipError(hipMemcpyHtoD(cosines, cosines_host.data(), ORDERS_NUM * sizeof(float)));
		// checkHipError(hipMemcpyHtoD(sines, sines_host.data(), ORDERS_NUM * sizeof(float)));
		tsinmemcpy.pause();
	}

	void setM()
	{
		tsetM.resume();
		log("set M");
		checkRocblasError(rocblas_scopy(handle, totalN * SIZE, M0_batched, 1, M_batched, 1));
		tsetM.pause();
	}

	void compute_next_step()
	{
		tnextStep.resume();
		log("Compute next step");
		checkRocblasError(rocblas_sgemv_strided_batched(handle,
			rocblas_operation_transpose, N, N, &delta_t, M_batched,
			N, totalN, dm_batched, 1, N, &one, dm_batched, 1, N, SIZE));
		tnextStep.pause();
	}

	void compute_sigma_minus()
	{
		tcompsigma.resume();
		log("compute sigma minus");
		checkRocblasError(rocblas_sgemv(handle,
			rocblas_operation_transpose, N, SIZE, &one, dm_batched,
			N, dot_re, 1, &zero, sigmas_re, 1));
		checkRocblasError(rocblas_sgemv(handle,
			rocblas_operation_transpose, N, SIZE, &one, dm_batched,
			N, dot_im, 1, &zero, sigmas_im, 1));
		tcompsigma.pause();
	}

	void compute(const vector<float>& tlist, const ModelParams& pars)
	{
		log("compute");
		for (int i = 0; i < LEN; i++)
		{
			setM();
			setMd1(pars, tlist[i]);
			setMd2(pars, tlist[i]);
			compute_next_step();
			compute_sigma_minus();
			add_to_fourier(tlist[i]);
		}
	}

	void add_to_fourier(float t)
	{
		taddtofft.resume();
		log("add_to_fourier");
		tsetsines.resume();
		set_sines_cosines(t);
		tsetsines.pause();
		tsger.resume();
		checkRocblasError(rocblas_sger(handle, SIZE, ORDERS_NUM,
			&delta_n, sigmas_re, 1, cosines, 1, fft_re, SIZE));
		checkRocblasError(rocblas_sger(handle, SIZE, ORDERS_NUM,
			&delta_n, sigmas_im, 1, sines, 1, fft_re, SIZE));
		checkRocblasError(rocblas_sger(handle, SIZE, ORDERS_NUM,
			&minus_delta_n, sigmas_re, 1, sines, 1, fft_im, SIZE));
		checkRocblasError(rocblas_sger(handle, SIZE, ORDERS_NUM,
			&delta_n, sigmas_im, 1, cosines, 1, fft_im, SIZE));
		tsger.pause();
		taddtofft.pause();
	}

	vector<complex<float>> get_fft_values() {
		using namespace std::complex_literals;
		log("get fft values");
		int FFT_SIZE = SIZE * ORDERS_NUM;
		vector<float> fft_re_host(FFT_SIZE);
		vector<float> fft_im_host(FFT_SIZE);
		vector<complex<float>> fft_host(FFT_SIZE);
		checkHipError(hipMemcpyDtoH(fft_re_host.data(), fft_re, fft_re_host.size() * sizeof(float)));
		checkHipError(hipMemcpyDtoH(fft_im_host.data(), fft_im, fft_im_host.size() * sizeof(float)));
		transform(fft_re_host.begin(), fft_re_host.end(), fft_im_host.begin(), fft_host.begin(),
			[](float re, float im) { return complex<float>(re, im); });
		return fft_host;
	} 
};

#endif // _SOLVER_
