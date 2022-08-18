/*
In comparison to other versions. Here the implicit runge-kutta method is applied. Also smartpointers from memory.h are used.
*/
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
#include <rocsolver/rocsolver.h>
#include "memory.h"
#include <fftw3.h>
#include <complex>
#include <string>
#include <algorithm>
#include <cstdarg>
#include <execution>
#include "timer.h"
using namespace std;

const int N { 15 };
const int totalN { N * N };

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

typedef std::shared_ptr<float> gpuvec;

inline auto newgpuvec(long long int numElements)
{
    return hip::shared<float>(numElements);
}

inline void togpu(vector<float>& host_v, gpuvec& gpu_v)
{
    checkHipError(hipMemcpyHtoD(gpu_v.get(), host_v.data(), host_v.size() * sizeof(float)));
}

inline void fromgpu(vector<float>& host_v, gpuvec& gpu_v)
{
    checkHipError(hipMemcpyDtoH(host_v.data(), gpu_v.get(), host_v.size() * sizeof(float)));
}

inline void gpufillzeros(gpuvec& gpu_v, long long int numElements)
{
    checkHipError(hipMemset(gpu_v.get(), 0, numElements * sizeof(float)));
}

void log(string&& str)
{
#ifdef __SDEBUG__
	cout << str << endl;
#endif
}

void log_hostvector(string&& name, vector<float>& m, int SIDE)
{
	cout << "Vector " << name << ":\n";
	cout.precision(2);
	for (int i = 0; i < m.size(); i++)
	{
        cout << m[i] << ' ';
	}
	cout << endl;
}

void log_hostmatrix(string&& name, vector<float>& m, int SIDE)
{
	cout << "Matrix " << name << ":\n";
	cout.precision(2);
	for (int i = 0; i < SIDE; i++)
	{
		for (int j = 0; j < SIDE; j++)
		{
			cout << m[i * SIDE + j] << '\t';
		}
		cout << endl;
	}
	cout << endl;
}


void log_gpuvector(string&& name, gpuvec& vec, int SIZE)
{
    vector<float> v(SIZE);
    fromgpu(v, vec);
	cout << "Vector " << name << ":\n";
    for (auto iter = v.begin(); iter != v.end(); iter++)
        cout << *iter << " ";
	cout << endl;
}

void log_gpumatrix(string&& name, gpuvec& mat, int SIDE)
{
    vector<float> m(SIDE * SIDE);
    fromgpu(m ,mat);
	cout << "Matrix " << name << ":\n";
	cout.precision(2);
	for (int i = 0; i < SIDE; i++)
	{
		for (int j = 0; j < SIDE; j++)
		{
			cout << m[i * SIDE + j] << '\t';
		}
		cout << endl;
	}
	cout << endl;
}

class Solver
{
// Host variables
	vector<float> M0_host;
	vector<float> Md1_host;
	vector<float> Md2_host;
	vector<float> M_host;
	vector<float> dm_host;
	vector<float> B0_host;
	vector<float> Bd1_host;
	vector<float> Bd2_host;
	float delta_t;
	float delta_n;
	const float one = 1;
	const float minusone = -1;
	const float zero = 0;
	int LEN;
	int ESIZE;
	int WSIZE;
	int SIZE;
	int ORDERS_NUM = 1;
    int METHOD_SIZE = 2;
    int Msize,
        Bsize,
        Mpartsize,
        Bpartsize;
	vector<float> Es_host;
	vector<float> Ws_host;
	vector<float> ones_host;
	vector<float> frequencies;
	vector<float> cosines_host;
	vector<float> sines_host;
    vector<float> butcher_a,
        butcher_b,
        butcher_c;
// GPU variables
	gpuvec M0;
	gpuvec M0_batched;
	gpuvec Md1;
	gpuvec Md2;
	gpuvec B0;
	gpuvec B0_batched;
	gpuvec Bd1;
	gpuvec Bd2;
	gpuvec M_batched;
	gpuvec B_batched;
	gpuvec dm;
	gpuvec dm_batched;
    gpuvec temp_dm;
    gpuvec K;
    gpuvec A;
    gpuvec A_batched;
	gpuvec dot_re;
	gpuvec dot_im;
	gpuvec sigmas_re;
	gpuvec sigmas_im;
	gpuvec Es;
	gpuvec Ws;
	gpuvec ones;
	gpuvec cosines;
	gpuvec sines;
	gpuvec dm_fft_cos;
	gpuvec dm_fft_sin;
	gpuvec fft_re;
	gpuvec fft_im;
    gpuvec method_ones;
	gpuvec identity_batched;
	std::shared_ptr<int> ipiv;
	std::shared_ptr<int> info;
	rocblas_handle handle;
// Debug variables
	// Timer tsetM;
	// Timer tsetMd1;
	// Timer tsetMd2;
	// Timer tnextStep;
	// Timer taddtofft;
	// Timer tcompsigma;
	// Timer tsetsines;
	// Timer tsger;
	// Timer tforsin;
	// Timer tsinmemcpy;

public:
	Solver(const ModelParams &pars, float dt, int length,
		const vector<float> &Wvals, const vector<float> &Evals,
		const vector<int> &orders) :
		M0_host(totalN, 0.f), Md1_host(totalN, 0.f), Md2_host(totalN, 0.f),
		M_host(totalN, 0.f), dm_host(N, 0.f), B0_host(N, 0.f),
		Bd1_host(N, 0.f), Bd2_host(N, 0.f),
		Es_host(Wvals.size() * Evals.size(), 0.f),
		Ws_host(Wvals.size() * Evals.size(), 0.f),
		ones_host(Wvals.size() * Evals.size(), 1.f),
		frequencies(orders.size(), 0.f),
		cosines_host(orders.size(), 0.f),
		sines_host(orders.size(), 0.f),
        butcher_a(METHOD_SIZE * METHOD_SIZE), butcher_b(METHOD_SIZE),
        butcher_c(METHOD_SIZE)
		// tsetM("setM"), tsetMd1("setMd1"), tsetMd2("setMd2"), tnextStep("compute_next_step"),
		// taddtofft("add_to_fourier"), tcompsigma("compute_sigma_minus"),
		// tsetsines("set_sines_cosines"), tsger("rocblas_sger"),
		// tforsin("for loop with sin"), tsinmemcpy("memcpy sin")
	{
    	checkRocblasError(rocblas_create_handle(&handle));
        rocblas_initialize();
    	checkRocblasError(rocblas_set_pointer_mode(handle, 
    		rocblas_pointer_mode_host));
		dm_host[N-1] = 1.f;
		delta_t = dt;
		LEN = length;
		delta_n = 1.f / static_cast<float>(LEN);
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
        Mpartsize = METHOD_SIZE * METHOD_SIZE * totalN;
        Bpartsize = METHOD_SIZE * N;
        Msize = SIZE * Mpartsize;
        Bsize = SIZE * Bpartsize;
		ORDERS_NUM = orders.size();
		log("Mallocs");
        M0 = newgpuvec(Mpartsize);
        M0_batched = newgpuvec(Msize);
        Md1 = newgpuvec(Mpartsize);
        Md2 = newgpuvec(Mpartsize);
        M_batched = newgpuvec(Msize);
        B0 = newgpuvec(Bpartsize);
        B0_batched = newgpuvec(Bsize);
        Bd1 = newgpuvec(Bpartsize);
        Bd2 = newgpuvec(Bpartsize);
        B_batched = newgpuvec(Bsize);
        dm = newgpuvec(N);
        dm_batched = newgpuvec(SIZE * N);
        temp_dm = newgpuvec(SIZE * N * METHOD_SIZE);
        method_ones = newgpuvec(METHOD_SIZE * SIZE);
        K = newgpuvec(Bsize);
        dot_re = newgpuvec(N);
        dot_im = newgpuvec(N);
        sigmas_re = newgpuvec(SIZE);
        sigmas_im = newgpuvec(SIZE);
        Es = newgpuvec(SIZE);
        Ws = newgpuvec(SIZE);
        ones = newgpuvec(SIZE);
        cosines = newgpuvec(ORDERS_NUM);
        sines = newgpuvec(ORDERS_NUM);
        dm_fft_cos = newgpuvec(N * SIZE * ORDERS_NUM);
        dm_fft_sin = newgpuvec(N * SIZE * ORDERS_NUM);
        fft_re = newgpuvec(SIZE * ORDERS_NUM);
        fft_im = newgpuvec(SIZE * ORDERS_NUM);
        A = newgpuvec(Mpartsize);
        A_batched = newgpuvec(Msize);
		identity_batched = newgpuvec(Msize);
		ipiv = hip::shared<int>(Bsize);
		info = hip::shared<int>(SIZE);
		log("Memcopy");
        togpu(Es_host, Es);
        togpu(Ws_host, Ws);
        togpu(ones_host, ones);
        vector<float> method_ones_host(SIZE * METHOD_SIZE, 1.f);
        togpu(method_ones_host, method_ones);
		log("FFT arrays set");
        gpufillzeros(dm_fft_cos, N * SIZE * ORDERS_NUM);
        gpufillzeros(dm_fft_sin, N * SIZE * ORDERS_NUM);
        gpufillzeros(dm_batched, N * SIZE);
        togpu(dm_host, dm);
		checkRocblasError(rocblas_sger(handle, N, SIZE, &one, dm.get(), 1,
			ones.get(), 1, dm_batched.get(), N));
		setM0(pars);
    	vector<float> dot_re_host(N, 0.f);
    	dot_re_host[0] = 1;
    	dot_re_host[10] = 1;
    	vector<float> dot_im_host(N, 0.f);
    	dot_im_host[3] = -1;
    	dot_im_host[13] = -1;
    	log("set dot_re and dot_im");
        togpu(dot_re_host, dot_re);
        togpu(dot_im_host, dot_im);
        set_method();
		make_identity_batched();
		setFrequencies(pars.domega, orders);
	}

	~Solver()
	{
		// tsetM.result();
		// tsetMd1.result();
		// tsetMd2.result();
		// tnextStep.result();
		// tcompsigma.result();
		// taddtofft.result();
		// tsetsines.result();
		// tsger.result();
		// tforsin.result();
		// tsinmemcpy.result();
		checkRocblasError(rocblas_destroy_handle(handle));
	}

    void set_method()
    {
        butcher_a = {0.25f, 0.25f - sqrt(3.f) / 6.f, 0.25f + sqrt(3.f) / 6.f, 0.25f};;
        vector<float> a_mat(Mpartsize, 0.f);
        int lda = METHOD_SIZE * N;
        for (int i = 0; i < METHOD_SIZE; i++)
        {
            for (int j = 0; j < METHOD_SIZE; j++)
            {
                float a = butcher_a[i * METHOD_SIZE + j];
                for (int k = 0; k < N; k++)
                    a_mat[j * totalN * METHOD_SIZE + i * N + k * (lda + 1)] = a;
            }
        }
        togpu(a_mat, A);
        gpufillzeros(A_batched, Msize);
		checkRocblasError(rocblas_sger(handle, Mpartsize, SIZE, &one, A.get(), 1, ones.get(), 1, A_batched.get(), Mpartsize));
        butcher_b = {0.5f, 0.5f};
        butcher_c = {0.5f - sqrt(3.f) / 6.f, 0.5f + sqrt(3.f) / 6.f};
    }

	void setM0(const ModelParams &pars)
	{
		log("set M0");
		float alphasqrt = pars.alpha * sqrt(pars.gamma1 * pars.gamma2);
		B0_host = {-pars.gamma1 - pars.gamma2 - pars.eta1, 0.f, 0.f, 0.f, 0.f, pars.gamma2, alphasqrt, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1, 0.f, 0.f, 0.f, 0.f, 0.f};
		M0_host = {-pars.gamma1 - pars.gamma2/2.f - pars.eta1, 0.f, 0.f, -pars.omega2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -alphasqrt, -pars.gamma1/2.f - pars.gamma2 - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.omega1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.gamma1/2.f - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.omega1 - pars.omega2, 0.f, 0.f, 0.f, pars.omega2, 0.f, 0.f, -pars.gamma1 - pars.gamma2/2.f - pars.eta1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.gamma1 - pars.gamma2 - pars.eta1, 0.f, 0.f, 0.f, 0.f, -pars.gamma2, 0.f, 0.f, 0.f, 0.f, -pars.gamma2, 0.f, 0.f, 0.f, 0.f, -2.f*alphasqrt, -pars.gamma1/2.f - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 0.f, -pars.omega1 + pars.omega2, -alphasqrt, 0.f, 0.f, 0.f, 0.f, -alphasqrt, alphasqrt, pars.gamma2, 0.f, 0.f, 0.f, 0.f, -pars.gamma1/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.omega1, 0.f, 0.f, 0.f, pars.omega1, 0.f, -alphasqrt, 0.f, 0.f, 0.f, -pars.gamma1/2.f - pars.gamma2 - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.omega1 - pars.omega2, 0.f, 0.f, -pars.gamma1/2.f - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.gamma1 - pars.eta1, -2.f*alphasqrt, 0.f, 0.f, 0.f, -pars.gamma1 - pars.gamma2 - pars.eta1, 0.f, 0.f, 0.f, 0.f, -pars.gamma1 - pars.eta1, pars.gamma1 + pars.eta1, alphasqrt, 0.f, 0.f, 0.f, 0.f, -alphasqrt, 0.f, 0.f, 0.f, -pars.gamma2/2.f, 0.f, 0.f, -pars.omega2, 0.f, 0.f, 0.f, pars.omega1 + pars.omega2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -pars.gamma1/2.f - pars.gamma2/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, alphasqrt, 0.f, 0.f, pars.omega1, pars.gamma2, 0.f, 0.f, 0.f, 0.f, -pars.gamma1/2.f - pars.eta1/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1, 0.f, 0.f, 0.f, alphasqrt, 0.f, 0.f, pars.omega2, 0.f, -alphasqrt, -pars.gamma2/2.f, 0.f, 0.f, 0.f, 0.f, 0.f, pars.gamma1 + pars.eta1, 2.f*alphasqrt, 0.f, 0.f, 0.f, pars.gamma2, 0.f, 0.f, 0.f, 0.f, 0.f};
        gpufillzeros(M0, Mpartsize);
        gpufillzeros(B0, Bpartsize);
        for (int i = 0; i < METHOD_SIZE; i++)
        {
            put_M_in_bigger_M(M0_host, M0, i);
            put_B_in_bigger_B(B0_host, B0, i);
        }
        gpufillzeros(M0_batched, Msize);
        gpufillzeros(B0_batched, Bsize);
		checkRocblasError(rocblas_sger(handle, Mpartsize, SIZE, &one, M0.get(), 1, ones.get(), 1, M0_batched.get(), Mpartsize));
		checkRocblasError(rocblas_sger(handle, Bpartsize, SIZE, &one, B0.get(), 1, ones.get(), 1, B0_batched.get(), Bpartsize));
	}

    void put_M_in_bigger_M(vector<float>& M_h, gpuvec& M_d, int pos)
    {
        checkRocblasError(rocblas_set_matrix(N, N, sizeof(float),
            M_h.data(), N, M_d.get() + pos * (totalN * METHOD_SIZE + N),
            N * METHOD_SIZE));
    }

    void put_B_in_bigger_B(vector<float>& B_h, gpuvec& B_d, int pos)
    {
        checkRocblasError(rocblas_set_vector(N, sizeof(float), B_h.data(),
            1, B_d.get() + pos * N, 1));
    }

	auto get_M0_host()
	{
		return M0_host;
	}

	void make_identity_batched()
	{
		vector<float> id_h(Mpartsize, 0.f);
		int n = N * METHOD_SIZE;
		for (int i = 0; i < Mpartsize; i += n + 1)
			id_h[i] = 1.f;
		auto id_d = newgpuvec(Mpartsize);
		togpu(id_h, id_d);
		checkRocblasError(rocblas_sger(handle, Mpartsize, SIZE, &one, id_d.get(), 1, ones.get(), 1, identity_batched.get(), Mpartsize));
	}

	void setMd1(const ModelParams &pars, float t)
	{
		// tsetMd1.resume();
		log("set Md1");
        gpufillzeros(Md1, Mpartsize);
        gpufillzeros(Bd1, Bpartsize);
        for (int i = 0; i < METHOD_SIZE; i++)
        {
            float t_ = t + butcher_c[i] * delta_t;
            float wsqrtcos = sqrt(pars.eta1) * cos(pars.domega * t_);
            float wsqrtsin = sqrt(pars.eta1) * sin(pars.domega * t_);
            Bd1_host = {0.f, 0.f, wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            Md1_host = {0.f, 0.f, -wsqrtcos, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, -2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, wsqrtcos, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 0.f, wsqrtsin, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, wsqrtcos, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, -2.f*wsqrtsin, 0.f, 0.f, wsqrtcos, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, 0.f, 0.f, -2.f*wsqrtsin, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, wsqrtsin, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtcos, 0.f, 0.f, wsqrtcos, 0.f, 0.f, wsqrtsin, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 0.f, wsqrtsin, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, wsqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, -wsqrtsin, 0.f, 0.f, wsqrtsin, 0.f, 0.f, -wsqrtcos, 0.f, 0.f, wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*wsqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*wsqrtsin, 0.f, 0.f};
            put_M_in_bigger_M(Md1_host, Md1, i);
            put_B_in_bigger_B(Bd1_host, Bd1, i);
        }
		checkRocblasError(rocblas_sger(handle, Mpartsize, SIZE, &one, 
			Md1.get(), 1, Ws.get(), 1, M_batched.get(), Mpartsize));
		checkRocblasError(rocblas_sger(handle, Bpartsize, SIZE, &one, 
			Bd1.get(), 1, Ws.get(), 1, B_batched.get(), Bpartsize));
		// tsetMd1.pause();
	}

	void setMd2(const ModelParams &pars, float t)
	{
		// tsetMd2.resume();
		log("set Md2");
        gpufillzeros(Md2, Mpartsize);
        gpufillzeros(Bd2, Bpartsize);
        for (int i = 0; i < METHOD_SIZE; i++)
        {
            float t_ = t + butcher_c[i] * delta_t;
		    float esqrtcos = sqrt(pars.gamma2) * cos(pars.domega * t_);
		    float esqrtsin = sqrt(pars.gamma2) * sin(pars.domega * t_);
		    Bd2_host = {0.f, esqrtcos, 0.f, 0.f, -esqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		    Md2_host = {0.f, 0.f, 0.f, 0.f, -2.f*esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, 0.f, -esqrtcos, 0.f, 0.f, -esqrtcos, 0.f, 0.f, -esqrtsin, 0.f, 0.f, esqrtsin, 0.f, 0.f, 0.f, 0.f,esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, esqrtsin, 0.f, 0.f, 0.f, 0.f, -esqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*esqrtsin, 0.f, 0.f, 0.f, 0.f, esqrtsin, 0.f, 0.f, 0.f, 0.f, esqrtsin, 2.f*esqrtcos, 0.f, 0.f, -2.f*esqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, -esqrtsin, 0.f, 0.f, 0.f, 0.f, esqrtsin, 0.f, 0.f, 0.f, 0.f,esqrtcos, 0.f, 0.f,esqrtcos, 0.f, 0.f, esqrtsin, 0.f, 0.f, -esqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, -esqrtsin, 0.f, 0.f, esqrtsin, 0.f, 0.f, -esqrtcos, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 0.f, 0.f, esqrtsin, 0.f, 0.f, 0.f, 0.f, -esqrtsin, esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.f*esqrtcos, 0.f, 0.f, 2.f*esqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, -esqrtsin, 0.f, 0.f, 0.f, 0.f, esqrtsin,esqrtcos, 0.f, 0.f, 0.f, 0.f, -esqrtcos, 0.f, 0.f, 0.f, 0.f, esqrtsin, 0.f, 0.f, -esqrtsin, 0.f, 0.f, esqrtcos, 0.f, 0.f,esqrtcos, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -esqrtsin, 0.f, 0.f, 0.f, 0.f, esqrtsin, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f*esqrtcos, 0.f, 0.f, -2.f*esqrtsin, 0.f};
            put_M_in_bigger_M(Md2_host, Md2, i);
            put_B_in_bigger_B(Bd2_host, Bd2, i);
        }
		checkRocblasError(rocblas_sger(handle, Mpartsize, SIZE, &one,
			Md2.get(), 1, Es.get(), 1, M_batched.get(), Mpartsize));
		checkRocblasError(rocblas_sger(handle, Bpartsize, SIZE, &one, 
			Bd2.get(), 1, Es.get(), 1, B_batched.get(), Bpartsize));
		// tsetMd2.pause();
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
		// tforsin.resume();
		for (int i = 0; i < ORDERS_NUM; i++)
		{
			cosines_host[i] = cos(frequencies[i] * t);
			sines_host[i] = sin(frequencies[i] * t);
		}
		// tforsin.pause();
		// tsinmemcpy.resume();
        togpu(cosines_host, cosines);
        togpu(sines_host, sines);
		// tsinmemcpy.pause();
	}

	void set_M_and_B(const ModelParams& pars, float t)
	{
		// tsetM.resume();
		log("set M");
		checkRocblasError(rocblas_scopy(handle, Msize, 
			M0_batched.get(), 1, M_batched.get(), 1));
		checkRocblasError(rocblas_scopy(handle, Bsize, 
			B0_batched.get(), 1, B_batched.get(), 1));
        setMd1(pars, t);
        setMd2(pars, t);
        checkRocblasError(rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none, Bpartsize, Bpartsize, Bpartsize, &one, M_batched.get(), Bpartsize, Mpartsize, A_batched.get(), Bpartsize, Mpartsize, &zero, M_batched.get(), Bpartsize, Mpartsize, SIZE));
		checkRocblasError(rocblas_saxpy(handle, Msize, &minusone, identity_batched.get(), 1, M_batched.get(), 1));
        append_Mrho_to_B();
		// tsetM.pause();
	}

    void append_Mrho_to_B()
    {
        gpufillzeros(temp_dm, Bsize);
        checkRocblasError(rocblas_sger_strided_batched(handle, N, METHOD_SIZE, &one, dm_batched.get(), 1, N, method_ones.get(), 1, METHOD_SIZE, temp_dm.get(), N, Bpartsize, SIZE));
        checkRocblasError(rocblas_sgemv_strided_batched(handle, rocblas_operation_transpose, Bpartsize, Bpartsize, &one, M_batched.get(), Bpartsize, Mpartsize, temp_dm.get(), 1, Bpartsize, &one, B_batched.get(), 1, Bpartsize, SIZE));

    }

	void compute_next_step()
	{
		// tnextStep.resume();
		log("Compute next step");
		checkRocblasError(rocblas_sgemv_strided_batched(handle,
			rocblas_operation_transpose, N, N, &delta_t, M_batched.get(),
			N, totalN, dm_batched.get(), 1, N, &one, dm_batched.get(), 1, 
			N, SIZE));
		checkRocblasError(rocblas_saxpy(handle, N * SIZE, &delta_t, B_batched.get(), 1, dm_batched.get(), 1));
		// tnextStep.pause();
	}

	void compute_sigma_minus()
	{
		// tcompsigma.resume();
		log("compute sigma minus");
		checkRocblasError(rocblas_sgemv(handle,
			rocblas_operation_transpose, N, SIZE * ORDERS_NUM,
			&one, dm_fft_cos.get(), N, dot_re.get(), 1, &zero, fft_re.get(), 1));
		checkRocblasError(rocblas_sgemv(handle,
			rocblas_operation_transpose, N, SIZE * ORDERS_NUM,
			&one, dm_fft_sin.get(), N, dot_im.get(), 1, &one, fft_re.get(), 1));
		checkRocblasError(rocblas_sgemv(handle,
			rocblas_operation_transpose, N, SIZE * ORDERS_NUM,
			&minusone, dm_fft_sin.get(), N, dot_re.get(), 1, &zero, fft_im.get(), 1));
		checkRocblasError(rocblas_sgemv(handle,
			rocblas_operation_transpose, N, SIZE * ORDERS_NUM,
			&one, dm_fft_cos.get(), N, dot_im.get(), 1, &one, fft_im.get(), 1));
		// tcompsigma.pause();
	}
	
	void solve_for_k()
	{
		checkRocblasError(rocsolver_sgesv_strided_batched(handle, Bpartsize, 1, M_batched.get(), Bpartsize, Mpartsize, ipiv.get(), Bpartsize, B_batched.get(), Bpartsize, Bpartsize, info.get(), SIZE));
	}

	void compute(const vector<float>& tlist, const ModelParams& pars)
	{
		log("compute");
        set_M_and_B(pars, 0.f);
		solve_for_k();
		// for (int i = 0; i < LEN; i++)
		// {
		// 	compute_k1(tlist[i], pars);
		// 	compute_k2(tlist[i], pars);
		// 	compute_k3(tlist[i], pars);
		// 	compute_k4(tlist[i], pars);
		// 	update_dm();
		// 	add_to_fourier(tlist[i]);
		// }
		// compute_sigma_minus();
	}

	// auto compute_with_dmsave(const vector<float>& tlist, const ModelParams& pars)
	// {
	// 	log("compute");
	// 	vector<float> dms{};
	// 	for (int i = 0; i < LEN; i++)
	// 	{
	// 		compute_k1(tlist[i], pars);
	// 		compute_k2(tlist[i], pars);
	// 		compute_k3(tlist[i], pars);
	// 		compute_k4(tlist[i], pars);
	// 		update_dm();
	// 		save_dm(dms);
	// 		add_to_fourier(tlist[i]);
	// 	}
	// 	compute_sigma_minus();
	// 	return dms;
	// }

    // void compute_k1(float t, const ModelParams& pars)
    // {
    //     setM();
    //     setMd1(pars, t);
    //     setMd2(pars, t);
    //     comp_M_dm_plus_B(dm_batched, k1);
    // }

    // void compute_k2(float t, const ModelParams& pars)
    // {
    //     float coeff1 = delta_t * 0.4f;
    //     setM();
    //     setMd1(pars, t + coeff1);
    //     setMd2(pars, t + coeff1);
    //     make_tmp_dm(&coeff1, k1);
    //     comp_M_dm_plus_B(temp_dm, k2);
    // }

    // void compute_k3(float t, const ModelParams& pars)
    // {
    //     float coeff1 = delta_t * 0.45573725f;
    //     float coeff2 = delta_t * 0.29697761f;
    //     float coeff3 = delta_t * 0.15875964f;
    //     setM();
    //     setMd1(pars, t + coeff1);
    //     setMd2(pars, t + coeff1);
    //     checkRocblasError(rocblas_scopy(handle, N * SIZE, dm_batched.get(), 1, temp_dm.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff2, k1.get(), 1, temp_dm.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff3, k2.get(), 1, temp_dm.get(), 1));
    //     comp_M_dm_plus_B(temp_dm, k3);
    // }
    
    // void compute_k4(float t, const ModelParams& pars)
    // {
    //     float coeff1 = delta_t * 0.21810040.f;
    //     float coeff2 = delta_t * -3.05096516f;
    //     float coeff3 = delta_t * 3.83286476f;
    //     setM();
    //     setMd1(pars, t + delta_t);
    //     setMd2(pars, t + delta_t);
    //     checkRocblasError(rocblas_scopy(handle, N * SIZE, dm_batched.get(), 1, temp_dm.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff1, k1.get(), 1, temp_dm.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff2, k2.get(), 1, temp_dm.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff3, k3.get(), 1, temp_dm.get(), 1));
    //     comp_M_dm_plus_B(temp_dm, k4);
    // }

    // void update_dm()
    // {
    //     float coeff1 = delta_t * 0.17476028f;
    //     float coeff2 = delta_t * -0.55148066f;
    //     float coeff3 = delta_t * 1.20553560.f;
    //     float coeff4 = delta_t * 0.17118478f;
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff1, k1.get(), 1, dm_batched.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff2, k2.get(), 1, dm_batched.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff3, k3.get(), 1, dm_batched.get(), 1));
    //     checkRocblasError(rocblas_saxpy(handle, N * SIZE, &coeff4, k4.get(), 1, dm_batched.get(), 1));
    // }

    void make_tmp_dm(const float* coeff, gpuvec& ki)
    {
        checkRocblasError(rocblas_scopy(handle, N * SIZE, dm_batched.get(), 1, temp_dm.get(), 1));
        checkRocblasError(rocblas_saxpy(handle, N * SIZE, coeff, ki.get(), 1, temp_dm.get(), 1));
    }

    void comp_M_dm_plus_B(gpuvec& dm_, gpuvec& out_)
    {
        checkRocblasError(rocblas_sgemv_strided_batched(handle,rocblas_operation_transpose, N, N, &one, M_batched.get(), N, totalN, dm_.get(), 1, N, &zero, out_.get(), 1, N, SIZE));
        checkRocblasError(rocblas_saxpy(handle, N * SIZE, &one, B_batched.get(), 1, out_.get(), 1));
    }

	void add_to_fourier(float t)
	{
		// taddtofft.resume();
		log("add_to_fourier");
		// tsetsines.resume();
		set_sines_cosines(t);
		// tsetsines.pause();
		// tsger.resume();
		checkRocblasError(rocblas_sger(handle, SIZE * N, ORDERS_NUM,
			&one, dm_batched.get(), 1, cosines.get(), 1, dm_fft_cos.get(), SIZE * N));
		checkRocblasError(rocblas_sger(handle, SIZE * N, ORDERS_NUM,
			&one, dm_batched.get(), 1, sines.get(), 1, dm_fft_sin.get(), SIZE * N));
		// tsger.pause();
		// taddtofft.pause();
	}

	auto get_dm_fft_cos()
	{
		vector<float> dfc_h(SIZE * N * ORDERS_NUM);
        fromgpu(dfc_h, dm_fft_cos);
		return dfc_h;
	}

	auto get_dm_fft_sin()
	{
		vector<float> dfs_h(SIZE * N * ORDERS_NUM);
        fromgpu(dfs_h, dm_fft_sin);
		return dfs_h;
	}

	auto get_sines()
	{
		vector<float> x(ORDERS_NUM);
        fromgpu(x, sines);
		return x;
	}

	auto get_cosines()
	{
		vector<float> x(ORDERS_NUM);
        fromgpu(x, cosines);
		return x;
	}

    auto get_temp_dm()
    {
		vector<float> x(Bsize);
        fromgpu(x, temp_dm);
		return x;
    }

	void save_dm(vector<float>& dms)
	{
		vector<float> x(SIZE * N);
        fromgpu(x, dm_batched);
		dms.insert(dms.end(), x.begin(), x.end());
	}

	auto get_dot_re()
	{
		vector<float> x(N);
        fromgpu(x, dot_re);
		return x;
	}

	auto get_dot_im()
	{
		vector<float> x(N);
        fromgpu(x, dot_im);
		return x;
	}

	auto get_fft_re()
	{
		vector<float> x(SIZE * ORDERS_NUM);
        fromgpu(x, fft_re);
		return x;
	}

	auto get_fft_im()
	{
		vector<float> x(SIZE * ORDERS_NUM);
        fromgpu(x, fft_im);
		return x;
	}

    auto get_method_ones()
    {
		vector<float> x(SIZE * METHOD_SIZE);
        fromgpu(x, method_ones);
		return x;
    }

	auto get_fft_values()
	{
		using namespace std::complex_literals;
		log("get fft values");
		int FFT_SIZE = SIZE * ORDERS_NUM;
		vector<float> fft_re_host(FFT_SIZE);
		vector<float> fft_im_host(FFT_SIZE);
		vector<complex<float>> fft_host(FFT_SIZE);
        fromgpu(fft_re_host, fft_re);
        fromgpu(fft_im_host, fft_im);
		transform(fft_re_host.begin(), fft_re_host.end(),
			fft_im_host.begin(), fft_host.begin(),
			[](float re, float im) { return complex<float>(re, im); });
		return fft_host;
	}

	vector<float> get_dm_batched()
	{
		vector<float> dm_h(SIZE * N);
        fromgpu(dm_h, dm_batched);
		return dm_h;
	}

	vector<float> get_dm()
	{
		vector<float> dm_h(N);
        fromgpu(dm_h, dm);
		return dm_h;
	}

	auto get_M()
	{
		vector<float> M_h(Msize);
        fromgpu(M_h, M_batched);
		return M_h;
	}

    auto get_M0()
	{
		vector<float> M_h(Mpartsize);
        fromgpu(M_h, M0);
		return M_h;
	}

	auto get_M0_batched()
	{
		vector<float> M_h(Msize);
        fromgpu(M_h, M0_batched);
		return M_h;
	}

	auto get_Md1()
	{
		vector<float> M_h(Mpartsize);
        fromgpu(M_h, Md1);
		return M_h;
	}

	auto get_Md2()
	{
		vector<float> M_h(Mpartsize);
        fromgpu(M_h, Md2);
		return M_h;
	}

	auto get_ones()
	{
		vector<float> M_h(SIZE);
        fromgpu(M_h, ones);
		return M_h;
	}
};

#endif // _SOLVER_
