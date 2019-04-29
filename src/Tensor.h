#include <CL_TENSOR.h>
#include <map>
#include <string>

//TensorCL wrapper with automatic differentiation
class Tensor
{
public:
	enum OPERATION
	{
		NONE,
		ADD_T, SUBS_T, MUL_T, DIV_T, NEG,
		ADD_N, SUBS_N, MUL_N, DIV_N,
		SIN, COS, TAN, EXP, LOG, TANH, POW,
		SUM, MIN_M, MAX_M, MIN_N, MAX_N,
		MORE_M, LESS_M, MORE_N, LESS_N,
		TRANSPOSE, DOT, REPEAT, GET_INDEX, IF_COND
	};

	Tensor(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1, unsigned int w = 1);
	Tensor(cl_tensor param);
	Tensor(TensorCL& input, std::pair<int, int> parents = std::pair<int, int>(-1, -1), OPERATION op = NONE);
	Tensor(int id);
	Tensor(Tensor& x, float fill);

	~Tensor();

	Tensor(Tensor & X);
	Tensor(Tensor && X);

	Tensor& operator=(Tensor &X);
	Tensor& operator=(Tensor &&X);
	Tensor& operator=(float a);

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

	Tensor sin();
	Tensor cos();
	Tensor tan();
	Tensor exp();
	Tensor log();
	Tensor tanh();
	Tensor operator^(float y); //power

	Tensor sum();
	Tensor min(Tensor &X);
	Tensor max(Tensor &X);
	Tensor min(float y = 0.f);
	Tensor max(float y = 0.f);

	Tensor indicies(int dim);
	void reshape(int x = 1, int y = 1, int z = 1, int w = 1); //TODO
	Tensor transpose(int dim_a = 0, int dim_b = 1);
	Tensor repeat(int n = 1);

	Tensor _if(Tensor & _true, float _false);

	Tensor dot(Tensor &X); //dot product, last dimension of this and second to last dimension of X

	TensorCL& GetTensor();

	int ID();

protected:
	std::vector<int> FindChilds(int id);
	void RecursiveDestruction(int id);
	void init(TensorCL &X, std::pair<int, int> parents = std::pair<int, int>(-1, -1), OPERATION op = NONE);

	//the id of this element inside the tape
	int tape_id;
	std::vector<int> old_ids;
	bool copied;
};


void PrintTAPE(bool disp_value);
void PrintTensor(Tensor& a);


Tensor operator+(float x, Tensor& Y);
Tensor operator-(float x, Tensor& Y);
Tensor operator*(float x, Tensor& Y);
Tensor operator/(float x, Tensor& Y);
Tensor operator>(float x, Tensor& Y);
Tensor operator<(float x, Tensor& Y);

Tensor sin(Tensor& X);
Tensor cos(Tensor& X);
Tensor tan(Tensor& X);
Tensor exp(Tensor& X);
Tensor log(Tensor& X);
Tensor sum(Tensor& X);
Tensor tanh(Tensor& X);

Tensor min(Tensor &X, Tensor& Y);
Tensor max(Tensor &X, Tensor& Y);
Tensor min(Tensor& X, float y = 0.f);
Tensor max(Tensor& X, float y = 0.f);
Tensor min(float y, Tensor& X);
Tensor max(float y, Tensor& X);

Tensor dot(Tensor& X, Tensor& Y);
Tensor indicies(Tensor& X, int dim = 0);
Tensor repeat(Tensor& X, int n);
Tensor transpose(Tensor& X, int dim_a = 0, int dim_b = 1);

Tensor _if(Tensor& _cond, Tensor& _true, float _false);

//backpropagation class
class Gradient
{
public:
	Gradient(Tensor END);

	Gradient(Gradient& A);
	Gradient& operator=(Gradient &X);

	//derivative with respect to
	Tensor wrt(Tensor& X);

private:

	void AddDerivative(int pnode, int gnode);
	std::map<int, int> dydx;
};