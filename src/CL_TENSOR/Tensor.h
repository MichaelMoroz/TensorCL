#pragma once
#include <CL_TENSOR.h>
#include <map>
#include <string>

//TensorCL wrapper witch creates an operation tree for automatic differentiation
class Tensor
{
public:
	enum OPERATION
	{
		NONE, NOT,
		ADD_T, SUBS_T, MUL_T, DIV_T, NEG,
		ADD_N, SUBS_N, MUL_N, DIV_N,
		SIN, COS, TAN, EXP, LOG, TANH, POW,
		SUM, MIN_M, MAX_M, MIN_N, MAX_N,
		MORE_M, LESS_M, MORE_N, LESS_N,
		TRANSPOSE, DOT, REPEAT, GET_INDEX, IF_COND, CUT, EXPAND, RESHAPE
	};

	Tensor();
	Tensor(Size s, float fill = 0.f, bool rand = false);
	Tensor(unsigned int x, unsigned int y, unsigned int z = 1, unsigned int w = 1);
	Tensor(cl_tensor param);
	Tensor(TensorCL& input, std::pair<int, int> parents = std::pair<int, int>(-1, -1), OPERATION op = NONE);
	Tensor(int id);
	Tensor(Tensor& x, float fill);
	Tensor(TensorData & A);

	~Tensor();

	Tensor(const Tensor & X);
	Tensor(Tensor && X);

	Tensor& operator=(const Tensor &X);
	Tensor& operator=(Tensor &&X);
	Tensor& operator=(float a);

	template<typename T> Tensor& operator+=(T X);
	template<typename T> Tensor& operator-=(T X);
	template<typename T> Tensor& operator*=(T X);
	template<typename T> Tensor& operator/=(T X);

	Tensor operator+(Tensor &X);
	Tensor operator-(Tensor &X);
	Tensor operator*(Tensor &X);
	Tensor operator/(Tensor &X);

	Tensor operator+(float x);
	Tensor operator-(float x);
	Tensor operator-();
	Tensor operator*(float x);
	Tensor operator/(float x);

	Tensor operator>(Tensor &X);
	Tensor operator<(Tensor &X);
	Tensor operator>(float x);
	Tensor operator<(float x);

	float operator()(int i = 0, int j = 0, int k = 0, int m = 0);

	Tensor diag(float x = 1.f, float y = 0.f);
	Tensor random();
	Tensor sin();
	Tensor cos();
	Tensor tan();
	Tensor exp();
	Tensor log();
	Tensor tanh();
	Tensor pow(float y); //power
	int operator[] (int dim);

	Tensor sum();
	Tensor min(Tensor &X);
	Tensor max(Tensor &X);
	Tensor min(float y = 0.f);
	Tensor max(float y = 0.f);

	Tensor indicies(int dim);
	Tensor transpose(int dim_a = 0, int dim_b = 1);
	Tensor repeat(int n = 1);
	Tensor cut(int from, int to);

	Tensor _if(Tensor & _true, float _false);

	Tensor dot(Tensor &X); //dot product, last dimension of this and second to last dimension of X

	TensorCL& GetTensor();

	cl_tensor GetParam();
	Size GetSize();

	int ID();

protected:
	std::vector<int> FindChilds(int id);
	void RecursiveDestructionChilds(int id);
	void RecursiveDestructionParents(int id);
	bool AreAllChildsDestroyed(int id);
	void Destroy(int id);
	void init(TensorCL &X, std::pair<int, int> parents = std::pair<int, int>(-1, -1), OPERATION op = NONE);

	//the id of this element inside the tape
	int tape_id;
	std::vector<int> old_ids;
};

void PrintTAPE(bool disp_value);
int TAPE_SIZE();

Tensor reshape(Tensor X, int x, int y, int z, int w);
Tensor reshape(Tensor X, cl_tensor x);

//backpropagation class
class Gradient
{
public:
	Gradient(Tensor END);
	Gradient(int tensor_id);
	Gradient(Gradient& A);
	Gradient& operator=(Gradient &X);

	//derivative with respect to
	Tensor& wrt(Tensor& X);
	Tensor& wrt(int tensor_id);

	~Gradient();
protected:
	void VJP(int outgrad_id, int out_id, Tensor::OPERATION op);
	void AddDerivative(int pnode, Tensor gnode);
	std::map<int, Tensor> dydx;
};

template<typename T>
inline Tensor & Tensor::operator+=(T X)
{
	return *this = *this + X;
}

template<typename T>
inline Tensor & Tensor::operator-=(T X)
{
	return *this = *this - X;
}

template<typename T>
inline Tensor & Tensor::operator*=(T X)
{
	return *this = *this * X;
}

template<typename T>
inline Tensor & Tensor::operator/=(T X)
{
	return *this = *this / X;
}
