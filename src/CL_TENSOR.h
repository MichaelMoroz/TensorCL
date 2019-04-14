#pragma once
#include <SFML_plot.h>
#include <CLFunction.h>
#include <iomanip>

using namespace std;

class CLMatrix
{
public:
	int M, N;
	OpenCL* CL;
	cl::Buffer buffer_matrix, buffer_info;
	cl::NDRange global;

	CLMatrix()
	{

	}

	//intitialize constant matrix
	CLMatrix(int A, int B, float X, OpenCL *cl): M(A), N(B)
	{
		CL = cl;

		buffer_matrix = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, M*N*sizeof(float));
		buffer_info = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, sizeof(float) * 20);

		global = cl::NDRange(M, N);

		float data[] = {M, N};
		CL->queue.enqueueWriteBuffer(buffer_info, CL_TRUE, 0, sizeof(data), data);

		cl::Kernel copy(CL->default_program, "set_kernel");
		copy.setArg(0, buffer_matrix);
		copy.setArg(1, sizeof(float), (void*)&X);
		copy.setArg(2, buffer_info);

		CL->queue.enqueueNDRangeKernel(copy, cl::NullRange, global, cl::NullRange);

		CL->queue.finish();
	}

	void operator=(float a)
	{
		cl::Kernel copy(CL->default_program, "set_kernel");
		copy.setArg(0, buffer_matrix);
		copy.setArg(1, sizeof(float), (void*)&a);
		copy.setArg(2, buffer_info);

		CL->queue.enqueueNDRangeKernel(copy, cl::NullRange, global, cl::NullRange);

		CL->queue.finish();
	}


	CLMatrix(const CLMatrix& C) 
	{
		N = C.N;
		M = C.M;
		CL = C.CL;

		buffer_matrix = C.buffer_matrix;
		buffer_info = C.buffer_info;

		global = cl::NDRange(M, N);
	}

	//intialize matrix with an array
	CLMatrix(int A, int B, float** X, OpenCL *cl) : M(A), N(B)
	{
		CL = cl;
		buffer_matrix = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, M*N * sizeof(float));
		buffer_info = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, sizeof(float) * 20);

		global = cl::NDRange(M, N);

		float data[] = { M, N };
		CL->queue.enqueueWriteBuffer(buffer_info, CL_TRUE, 0, sizeof(data), data);

		float* D = new float[M*N];
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				D[j*M + i] = X[i][j];
			}
		}

		CL->queue.enqueueWriteBuffer(buffer_matrix, CL_TRUE, 0, M*N * sizeof(float), D);

		delete[] D;
	}
	
	//empty initialization
	void e_init(int A, int B, OpenCL *cl)
	{
		M = A;
		N = B;

		CL = cl;

		buffer_matrix = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, M*N * sizeof(float));
		buffer_info = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, sizeof(float) * 20);

		global = cl::NDRange(M, N);

		float data[] = { M, N };
		CL->queue.enqueueWriteBuffer(buffer_info, CL_TRUE, 0, sizeof(data), data);
	}

	void operator =(CLMatrix&);

	void getdata(float**& A)
	{
		float* B = new float[N*M];
		CL->queue.enqueueReadBuffer(buffer_matrix, CL_TRUE, 0, M*N* sizeof(float), B);

		A = new float*[M];
		for (int i = 0; i < M; i++)
		{
			A[i] = new float[N];
			for (int j = 0; j < N; j++)
			{
				A[i][j] = B[j*M + i];
			}
		}

		delete[] B;
	}

	void loaddata(float** A)
	{
		float* D = new float[M*N];
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				D[j*M + i] = A[i][j];
			}
		}

		CL->queue.enqueueWriteBuffer(buffer_matrix, CL_TRUE, 0, M*N * sizeof(float), D);

		delete[] D;
	}

	~ CLMatrix()
	{

	}
};

void CLMatrix::operator=(CLMatrix& rhs)
{
	if (M != rhs.M || N != rhs.N)
	{	
		CL = rhs.CL;

		M = rhs.M;
		N = rhs.N;
		
		buffer_matrix = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, M*N * sizeof(float));
		buffer_info = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, sizeof(float) * 20);

		global = cl::NDRange(rhs.M, rhs.N);

		float data[] = { rhs.M, rhs.N };
		CL->queue.enqueueWriteBuffer(buffer_info, CL_TRUE, 0, sizeof(data), data);
	}
	
	cl::Kernel copy(CL->default_program, "copy_kernel");
	copy.setArg(0,buffer_matrix);
	copy.setArg(1,rhs.buffer_matrix);
	copy.setArg(2,buffer_info);

	CL->queue.enqueueNDRangeKernel(copy, cl::NullRange, global, cl::NullRange);

	CL->queue.finish();
}



void print_matrix(CLMatrix &A, int p)
{
	float **a;
	A.getdata(a);

	for (int i = 0; i < A.M; i++)
	{
		for (int j = 0; j < A.N; j++)
		{
			cout <<setprecision(p)<< a[i][j] << " ";
		}
		cout << endl;
	}
}

void print_matrix_txt(CLMatrix &A, int p, string fname)
{
	ofstream save(fname.c_str());
	save.clear();
	float **a;
	A.getdata(a);

	for (int i = 0; i < A.N; i++)
	{
		for (int j = 0; j < A.M; j++)
		{
			save << setprecision(p) << a[j][i] << " ";
		}
		save << endl;
	}

	save.close();
}


void print_matrix_cs(CLMatrix &A, int p)
{
	float **a;
	A.getdata(a);

	for (int i = 0; i < A.N; i++)
	{
		float bb = 0;
		for (int j = 0; j < A.M; j++)
		{
			bb += pow(a[j][i],2);
		}
		cout << sqrt(bb) << " ";
	}
	cout << endl;
}

void print_matrix_t(CLMatrix &A, int p)
{
	float **a;
	A.getdata(a);

	for (int i = 0; i < A.N; i++)
	{
		for (int j = 0; j < A.M; j++)
		{
			cout << setprecision(p) << a[j][i] << " ";
		}
		cout << endl;
	}
}

void print_matrix_column(CLMatrix &A, int C)
{
	float **a;
	A.getdata(a);

	for (int i = 0; i < A.M; i++)
	{
		cout << setprecision(2) << a[i][C] << " ";
		cout << endl;
	}
}


const int TS = 8;
cl::NDRange local(TS, TS);

//use minimal possible global range for the specified local range
cl::NDRange getGlobal(int M, int N)
{
	return cl::NDRange(ceil((float)M/ (float)TS)*TS, ceil((float)N / (float)TS)*TS);
}

void MatrixMult(CLMatrix& A, CLMatrix& B, CLMatrix& C)
{
	if (A.N == B.M && C.M == A.M &&  C.N == B.N)
	{
		cl::Kernel kernel(A.CL->default_program, "simple_mult");

		// Set the arguments of the myGEMM kernel
		kernel.setArg(0, A.buffer_matrix);
		kernel.setArg(1, A.buffer_info);
		kernel.setArg(2, B.buffer_matrix);
		kernel.setArg(3, B.buffer_info);
		kernel.setArg(4, C.buffer_matrix);

		A.CL->queue.enqueueNDRangeKernel(kernel, cl::NullRange, getGlobal(C.M,C.N), local);
		A.CL->queue.finish();
	}
	else
	{
		cout << "ERROR: Matrix multiplication dimension mismatch" << endl;
	}
}

void MatrixMultBias(CLMatrix& A, CLMatrix& B, CLMatrix& C)
{
	if (A.N == B.M+1 && C.M == A.M &&  C.N == B.N)
	{
		cl::Kernel kernel(A.CL->default_program, "simple_mult_bias");

		kernel.setArg(0, A.buffer_matrix);
		kernel.setArg(1, A.buffer_info);
		kernel.setArg(2, B.buffer_matrix);
		kernel.setArg(3, B.buffer_info);
		kernel.setArg(4, C.buffer_matrix);

		A.CL->queue.enqueueNDRangeKernel(kernel, cl::NullRange, getGlobal(C.M, C.N), local);
		A.CL->queue.finish();
	}
	else
	{
		cout << "ERROR: Matrix multiplication dimension mismatch" << endl;
	}
}

void MatrixMultBiasL(CLMatrix& A, CLMatrix& B, CLMatrix& C)
{
	if (A.N == B.M && C.M == A.M+1 &&  C.N == B.N)
	{
		cl::Kernel kernel(A.CL->default_program, "simple_mult_bias_l");

		kernel.setArg(0, A.buffer_matrix);
		kernel.setArg(1, A.buffer_info);
		kernel.setArg(2, B.buffer_matrix);
		kernel.setArg(3, B.buffer_info);
		kernel.setArg(4, C.buffer_matrix);

		A.CL->queue.enqueueNDRangeKernel(kernel, cl::NullRange, getGlobal(C.M, C.N), local);
		A.CL->queue.finish();
	}
	else
	{
		cout << "ERROR: Matrix multiplication dimension mismatch" << endl;
	}
}

void UseCLFunction(vector<CLMatrix> &A, string program_name)
{
	int N = A.size();
	if (N > 0)
	{
		cl::Kernel prog(A[0].CL->default_program,program_name.c_str());
		for (int i = 0; i < N; i++)
		{
			prog.setArg(i, A[i].buffer_matrix);
		}
		prog.setArg(N, A[0].buffer_info);

		A[0].CL->queue.enqueueNDRangeKernel(prog, cl::NullRange, getGlobal(A[0].M, A[0].N), local);
		A[0].CL->queue.finish();
	}
}

void UseCLFunction(CLMatrix &M, vector<cl::Buffer> &A, string program_name)
{
	int N = A.size();
	if (N > 0)
	{
		cl::Kernel prog(M.CL->default_program, program_name.c_str());
		
		for (int i = 0; i < N; i++)
		{
			prog.setArg(i, A[i]);
		}

		M.CL->queue.enqueueNDRangeKernel(prog, cl::NullRange, getGlobal(M.M, M.N), local);
		M.CL->queue.finish();
	}
}


void UseCLFunction(CLMatrix &M, vector<cl::Buffer> &A, vector<float> &B, string program_name)
{
	int N = A.size();
	int Ms = B.size();
	if (N > 0)
	{
		cl::Kernel prog(M.CL->default_program, program_name.c_str());

		for (int i = 0; i < N; i++)
		{
			prog.setArg(i, A[i]);
		}
		for (int j = 0; j < Ms; j++)
		{
			prog.setArg(N + j, sizeof(float), (void*)&B[j]);
		}

		M.CL->queue.enqueueNDRangeKernel(prog, cl::NullRange, getGlobal(M.M, M.N), local);
		M.CL->queue.finish();
	}
}


void UseCLFunction(vector<CLMatrix> &A, vector<float> &B, string program_name)
{
	int N = A.size();
	int M = B.size();
	if (N > 0)
	{
		cl::Kernel prog(A[0].CL->default_program, program_name.c_str());
		int i, j;
		for (i = 0; i < N; i++)
		{
			prog.setArg(i, A[i].buffer_matrix);
		}
		prog.setArg(N, A[0].buffer_info);

		for (j = 0; j < M; j++)
		{
			prog.setArg(N+j+1, sizeof(float), (void*)&B[j]);
		}

		A[0].CL->queue.enqueueNDRangeKernel(prog, cl::NullRange, getGlobal(A[0].M, A[0].N), local);
		A[0].CL->queue.finish();
	}
}



//sum with log(N) speed
float FastSum(CLMatrix &A)
{
	int Level = 1;
	int K = 8; //work per thread

	int NN = A.M*A.N;
	int iter = floor(log(NN) / log(K)) + 1;

	cl::Kernel prog(A.CL->default_program, "FastSum");

	prog.setArg(0, A.buffer_matrix);
	prog.setArg(1, A.buffer_info);
	prog.setArg(2, sizeof(int), (void*)&Level);
	prog.setArg(3, sizeof(int), (void*)&K);

	for (Level = 1; Level <= iter;)
	{
		A.CL->queue.enqueueNDRangeKernel(prog, cl::NullRange, cl::NDRange(floor(NN / powf(8, Level)) + 1), cl::NullRange);
		A.CL->queue.finish();
		Level++;
		prog.setArg(2, sizeof(int), (void*)&Level);
	}
	

	float a;
	A.CL->queue.enqueueReadBuffer(A.buffer_matrix, CL_TRUE, 0, sizeof(float), &a);

	return a;
}

float FastSumMod(CLMatrix &A)
{
	CLMatrix Buf = A;

	int Level = 1;
	int K = 8; //work per thread

	int NN = A.M*A.N;
	int iter = floor(log(NN) / log(K)) + 1;
	cl::Kernel prog(A.CL->default_program, "FastSumMod");


	prog.setArg(0, Buf.buffer_matrix);
	prog.setArg(1, Buf.buffer_info);
	prog.setArg(2, sizeof(int), (void*)&Level);
	prog.setArg(3, sizeof(int), (void*)&K);

	for (Level = 1; Level <= iter;)
	{
		A.CL->queue.enqueueNDRangeKernel(prog, cl::NullRange, cl::NDRange(floor(NN / powf(8, Level)) + 1), cl::NullRange);
		A.CL->queue.finish();
		Level++;
		prog.setArg(2, sizeof(int), (void*)&Level);
	}


	float a;
	A.CL->queue.enqueueReadBuffer(Buf.buffer_matrix, CL_TRUE, 0, sizeof(float), &a);

	return a;
}



void MatrixBenchmark(int S, OpenCL *CL)
{
	sf::Clock timer;
	CLMatrix A(S, S, 0.f, CL), B(S, S, 0.f, CL), C(S, S, 0.f, CL);

	MatrixMult(A, B, C);

	float time_elapsed = timer.getElapsedTime().asMicroseconds()/1000.f;
	std::cout << S << "x" << S << " multiplication time = " << time_elapsed << " ms" << endl;
}

void MatrixStressTest(int S, float time, OpenCL *CL)
{
	sf::Clock timer;
	CLMatrix A(S, S, 0.f, CL), B(S, S, 0.f, CL), C(S, S, 0.f, CL);

	int i = 0;
	while (timer.getElapsedTime().asSeconds() < time)
	{
		i++;
		MatrixMult(A, B, C);
	}
	cout << i << " " << S << "x" << S << " multiplications in " << time << " sec" << endl;
	cout << 100*4*time/i << endl;
}


void Randomize_Matrix(CLMatrix &A, float d)
{
	cl::Kernel kernel1(A.CL->default_program, "Randomize");
	float a = -d, b = d;
	float seed = rand();

	kernel1.setArg(0, sizeof(float), (void*)&a);
	kernel1.setArg(1, sizeof(float), (void*)&b);
	kernel1.setArg(2, sizeof(float), (void*)&seed);
	kernel1.setArg(3, A.buffer_matrix);
	kernel1.setArg(4, A.buffer_info);

	A.CL->queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(A.M, A.N), cl::NullRange);
	A.CL->queue.finish();
}


void RandomColumnMixer(CLMatrix& A, int S, int sd)
{
	cl::Kernel kernel1(A.CL->default_program, "Mixer");
	srand(sd);
	kernel1.setArg(0, A.buffer_matrix);
	kernel1.setArg(1, sizeof(float), (void*)&A.M);
	kernel1.setArg(2, sizeof(float), (void*)&A.N);

	for (int i = 0; i < S; i++)
	{
		int seed = rand();
		int lvl = floor(rand() % ((int)A.N / 2 - 1));
		kernel1.setArg(3, sizeof(int), (void*)&lvl);
		kernel1.setArg(4, sizeof(int), (void*)&seed);
		A.CL->queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(A.M, ceil(A.N / 2.f)), cl::NullRange);
	}
	A.CL->queue.finish();
}