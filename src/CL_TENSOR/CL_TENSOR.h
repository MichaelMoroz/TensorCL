#pragma once
#include <CLFunction.h>
#include <vector>
#include <iomanip>

using namespace std;

constexpr auto MAX_DIM = 8;
constexpr auto TS = 8;

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
	TensorCL(int x, int y, int z = 1, int w = 1); 
	TensorCL(cl_tensor p);

	TensorCL(const TensorCL & X);
	TensorCL(TensorCL & X, float fill);
	TensorCL(TensorCL&& X);

	TensorCL& operator=(const TensorCL &X);
	TensorCL& operator=(TensorCL&& X);
	TensorCL& operator=(float a);

	TensorCL& operator+=(TensorCL &X);
	TensorCL& operator-=(TensorCL &X);
	TensorCL& operator*=(TensorCL &X);
	TensorCL& operator/=(TensorCL &X);

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

	float operator()(int i = 1, int j = 1, int k = 1, int m = 1);

	TensorCL random();
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

	TensorCL();

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

template<typename T> void PrintTensor(T& a);

void PrintTensor(TensorCL& a);

template<typename T> T operator+(float x, T& Y);
template<typename T> T operator-(float x, T& Y);
template<typename T> T operator*(float x, T& Y);
template<typename T> T operator/(float x, T& Y);
template<typename T> T operator>(float x, T& Y);
template<typename T> T operator<(float x, T& Y);

template<typename T> T random(T& X);
template<typename T> T sin(T& X);
template<typename T> T cos(T& X);
template<typename T> T tan(T& X);
template<typename T> T exp(T& X);
template<typename T> T log(T& X);
template<typename T> T sum(T& X);
template<typename T> T tanh(T& X);

template<typename T> T min(T &X, T& Y);
template<typename T> T max(T &X, T& Y);
template<typename T> T min(T& X, float y = 0.f);
template<typename T> T max(T& X, float y = 0.f);
template<typename T> T min(float y, T& X);
template<typename T> T max(float y, T& X);

template<typename T> T dot(T& X, T& Y);
template<typename T> T indicies(T& X, int dim = 0);
template<typename T> T repeat(T& X, int n = 1);
template<typename T> T transpose(T& X, int dim_a = 0, int dim_b = 1);

template<typename T> T _if(T& _cond, T& _true, T& _false);
template<typename T> T _if(T& _cond, T& _true, float _false);
template<typename T> T _if(T& _cond, float _true, T& _false);
template<typename T> T _if(T& _cond, float _true, float _false);

template<typename T>
inline void PrintTensor(T & a)
{
	PrintTensor(a.GetTensor());
}

template<typename T>
inline T operator+(float x, T & Y)
{
	return Y + x;
}

template<typename T>
inline T operator-(float x, T & Y)
{
	return -Y + x;
}

template<typename T> 
inline T operator*(float x, T& Y)
{
	return Y * x;
}

template<typename T>
inline T operator/(float x, T& Y)
{
	return Y / x;
}

template<typename T> 
inline T operator>(float x, T& Y)
{
	return Y < x;
}

template<typename T> 
inline T operator<(float x, T& Y)
{
	return Y > x;
}

template<typename T>
inline T random(T& X)
{
	return X.random();
}

template<typename T> 
inline T sin(T& X)
{
	return X.sin();
}

template<typename T> 
inline T cos(T& X)
{
	return X.cos();
}

template<typename T> 
inline T tan(T& X)
{
	return X.tan();
}

template<typename T> 
inline T exp(T& X)
{
	return X.exp();
}

template<typename T> 
inline T log(T& X)
{
	return X.log();
}

template<typename T> 
inline T sum(T& X)
{
	return X.sum();
}

template<typename T>
inline T tanh(T & X)
{
	return X.tanh();
}

template<typename T>
inline T min(T & X, T & Y)
{
	return X.min(Y);
}

template<typename T>
inline T max(T & X, T & Y)
{
	return X.max(Y);
}

template<typename T>
inline T min(T & X, float y)
{
	return X.min(y);
}

template<typename T>
inline T max(T & X, float y)
{
	return X.max(y);
}

template<typename T>
inline T min(float y, T & X)
{
	return X.max(y);
}

template<typename T>
inline T max(float y, T & X)
{
	return X.min(y);
}

template<typename T>
inline T dot(T & X, T & Y)
{
	return X.dot(Y);
}

template<typename T>
inline T indicies(T & X, int dim)
{
	return  X.indicies(dim);
}

template<typename T>
inline T repeat(T & X, int n)
{
	return X.repeat(n);
}

template<typename T>
inline T transpose(T & X, int dim_a, int dim_b)
{
	return X.transpose(dim_a, dim_b);
}

template<typename T>
inline T _if(T & _cond, T & _true, T & _false)
{
	return _cond._if(_true, _false);
}

template<typename T>
inline T _if(T & _cond, T & _true, float _false)
{
	return _cond._if(_true, _false);
}

template<typename T>
inline T _if(T & _cond, float _true, T & _false)
{
	return _cond._if(_true, _false);
}

template<typename T>
inline T _if(T & _cond, float _true, float _false)
{
	return _cond._if(_true, _false);
}
