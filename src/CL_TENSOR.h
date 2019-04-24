#pragma once
#include <SFML_plot.h>
#include <CLFunction.h>
#include <vector>
#include <iomanip>

using namespace std;

#define MAX_DIM 8
#define TS 8

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
void TensorUseOpenCL(OpenCL* cl);

class TensorCL
{
public:
	TensorCL(int r, vector<int> s); //General tensor init
	TensorCL(int x); //vector
	TensorCL(int x, int y); //matrix
	TensorCL(int x, int y, int z); //3d matrix
	TensorCL(int x, int y, int z, int w); //4d matrix
	TensorCL(cl_tensor p);

	TensorCL(TensorCL & X);
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

	TensorCL sin();
	TensorCL cos();
	TensorCL tan();
	TensorCL exp();
	TensorCL log();
	TensorCL operator^(float y); //power

	TensorCL sum();
	TensorCL min(TensorCL &X);
	TensorCL max(TensorCL &X);
	TensorCL min(float y = 0.f);
	TensorCL max(float y = 0.f);

	TensorCL indicies(int dim);
	void reshape(int x = 1, int y = 1, int z = 1, int w = 1);
	TensorCL transpose(int dim_a = 0, int dim_b = 1);

	TensorCL dot(TensorCL &X); //dot product, last dimension of this and second to last dimension of X
	
	void LoadData(float* data_ptr);
	float* GetData();
	int GetLength();
	cl_tensor GetParam();

	void release();

	~TensorCL();

private:
	void init_data();
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

TensorCL sin(TensorCL& X);
TensorCL cos(TensorCL& X);
TensorCL tan(TensorCL& X); 
TensorCL exp(TensorCL& X);
TensorCL log(TensorCL& X);
TensorCL sum(TensorCL& X);

TensorCL min(TensorCL &X, TensorCL& Y);
TensorCL max(TensorCL &X, TensorCL& Y);
TensorCL min(TensorCL& X, float y = 0.f);
TensorCL max(TensorCL& X, float y = 0.f);
TensorCL min(float y, TensorCL& X);
TensorCL max(float y, TensorCL& X);

TensorCL dot(TensorCL& X, TensorCL& Y);
TensorCL indicies(TensorCL& X, int dim = 0);
TensorCL transpose(TensorCL& X, int dim_a = 0, int dim_b = 1);