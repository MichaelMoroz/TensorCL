#pragma once
#include <CLFunction.h>
#include <vector>
#include <iomanip>

using namespace std;

#define MAX_DIM 8
#define TS 8

#undef min
#undef max

#pragma pack(push, r1, 1)
typedef struct
{
	cl_int size[MAX_DIM] = {1};
	cl_int rank = 1;
	cl_int length = 1;
} cl_tensor;
#pragma pack(pop, r1)

#pragma pack(push, r1, 1)
typedef struct
{
	cl_int index[MAX_DIM] = { 1 };
} cl_index;
#pragma pack(pop, r1)

bool AreTensorsEqual(cl_tensor x, cl_tensor y);
bool AreTensorsCompatible(cl_tensor x, cl_tensor y);
cl_tensor TensorDotResult(cl_tensor x, cl_tensor y);

cl_tensor Transpose(cl_tensor x, int dim_a, int dim_b);
cl_tensor GetSumTensor(cl_tensor x);
cl_tensor Repeat(cl_tensor x, int n); 
void TensorUseOpenCL(OpenCL* cl);

class TensorCL
{
public:
	TensorCL(int r, vector<int> s); 
	TensorCL(int x = 1, int y = 1, int z = 1, int w = 1); 
	TensorCL(cl_tensor p);

	TensorCL(TensorCL & X);
	TensorCL(TensorCL & X, float fill);
	TensorCL(TensorCL&& X);

	TensorCL& operator=(TensorCL &X);
	TensorCL& operator=(TensorCL&& X);
	TensorCL& operator=(float a);

	TensorCL operator+(TensorCL &X);
	TensorCL operator-(TensorCL &X);
	TensorCL operator*(TensorCL &X);
	TensorCL operator/(TensorCL &X);


	TensorCL operator+(float x);
	TensorCL operator-(float x);
	TensorCL operator-();
	TensorCL operator*(float x);
	TensorCL operator/(float x);

	TensorCL operator>(TensorCL &X);
	TensorCL operator<(TensorCL &X);
	TensorCL operator>(float x);
	TensorCL operator<(float x);

	TensorCL sin();
	TensorCL cos();
	TensorCL tan();
	TensorCL exp();
	TensorCL log();
	TensorCL tanh();
	TensorCL operator^(float y); //power

	TensorCL sum();
	TensorCL min(TensorCL &X);
	TensorCL max(TensorCL &X);
	TensorCL min(float y = 0.f);
	TensorCL max(float y = 0.f);

	TensorCL _if (TensorCL & _true, TensorCL & _false);
	TensorCL indicies(int dim);
	void reshape(int x = 1, int y = 1, int z = 1, int w = 1); //TODO
	TensorCL transpose(int dim_a = 0, int dim_b = 1);
	TensorCL repeat(int n);

	TensorCL dot(TensorCL &X); //dot product, last dimension of this and second to last dimension of X
	
	void LoadData(float* data_ptr);
	float* GetData();
	int GetLength();
	cl_tensor GetParam();

	void release();

	~TensorCL();

protected:
	void init_data(float value = 0.f);
	TensorCL MAD(float a, float b); //multiplication and addition

	cl_tensor param;
	
	//OpenCL stuff
	cl_mem data;

	float* host_data;
};

void PrintTensor(TensorCL& a);

TensorCL operator+(float x, TensorCL& Y);
TensorCL operator-(float x, TensorCL& Y);
TensorCL operator*(float x, TensorCL& Y);
TensorCL operator/(float x, TensorCL& Y);
TensorCL operator>(float x, TensorCL& Y);
TensorCL operator<(float x, TensorCL& Y);

TensorCL sin(TensorCL& X);
TensorCL cos(TensorCL& X);
TensorCL tan(TensorCL& X); 
TensorCL exp(TensorCL& X);
TensorCL log(TensorCL& X);
TensorCL sum(TensorCL& X);
TensorCL tanh(TensorCL& X);

TensorCL min(TensorCL &X, TensorCL& Y);
TensorCL max(TensorCL &X, TensorCL& Y);
TensorCL min(TensorCL& X, float y = 0.f);
TensorCL max(TensorCL& X, float y = 0.f);
TensorCL min(float y, TensorCL& X);
TensorCL max(float y, TensorCL& X);

TensorCL dot(TensorCL& X, TensorCL& Y);
TensorCL indicies(TensorCL& X, int dim = 0);
TensorCL repeat(TensorCL& X, int n = 1);
TensorCL transpose(TensorCL& X, int dim_a = 0, int dim_b = 1);

TensorCL _if(TensorCL& _cond, TensorCL& _true, TensorCL& _false);
TensorCL _if(TensorCL& _cond, TensorCL& _true, float _false);
TensorCL _if(TensorCL& _cond, float _true, TensorCL& _false);
TensorCL _if(TensorCL& _cond, float _true, float _false);