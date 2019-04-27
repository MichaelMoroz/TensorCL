#pragma once
#include <Tensor.h>
#include <vector>

int idt = 0;
// operation trees/recording tape
// only one instance exists
std::map<int, TensorCL> VALUE_TAPE;
std::map<int, Tensor::OPERATION> OPERATION_TAPE;
std::map<int, std::pair<int, int> > PARENTS_TAPE;
std::map<int, float> FLOAT_TAPE;
std::map<int, std::pair<int, int> > TRANSPOSE_TAPE;
std::map<int, int> REPEAT_TAPE;

Tensor::Tensor(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
	init(TensorCL(x, y, z, w));
}

Tensor::Tensor(cl_tensor param)
{
	init(TensorCL(param));
}

Tensor::Tensor(TensorCL & input, std::pair<int, int> parents, OPERATION op)
{
	init(input, parents, op);
}

Tensor::Tensor(int id)
{
	tape_id = id;
}

Tensor::Tensor(Tensor & x, float fill)
{
	Tensor(TensorCL(VALUE_TAPE[x.tape_id], fill));
}

void Tensor::init(TensorCL & X, std::pair<int, int> parents, OPERATION op)
{
	tape_id = idt;
	VALUE_TAPE[idt] = X;
	OPERATION_TAPE.emplace(idt, op);
	PARENTS_TAPE.emplace(idt, parents);
	idt++;
}

Tensor Tensor::sin()
{
	return Tensor(VALUE_TAPE[this->tape_id].sin(), std::pair<int, int>(this->tape_id, -1), SIN);
}

Tensor Tensor::cos()
{
	return Tensor(VALUE_TAPE[this->tape_id].cos(), std::pair<int, int>(this->tape_id, -1), COS);
}

Tensor Tensor::tan()
{
	return Tensor(VALUE_TAPE[this->tape_id].tan(), std::pair<int, int>(this->tape_id, -1), TAN);
}

Tensor Tensor::exp()
{
	return Tensor(VALUE_TAPE[this->tape_id].exp(), std::pair<int, int>(this->tape_id, -1), EXP);
}

Tensor Tensor::log()
{
	return Tensor(VALUE_TAPE[this->tape_id].log(), std::pair<int, int>(this->tape_id, -1), LOG);
}

Tensor Tensor::tanh()
{
	return Tensor(VALUE_TAPE[this->tape_id].tanh(), std::pair<int, int>(this->tape_id, -1), TANH);
}

Tensor Tensor::operator^(float y)
{
	FLOAT_TAPE[idt] = y;
	return Tensor(VALUE_TAPE[this->tape_id]^y, std::pair<int, int>(this->tape_id, -1), POW);
}

Tensor Tensor::sum()
{
	return Tensor(VALUE_TAPE[this->tape_id].sum(), std::pair<int, int>(this->tape_id, -1), SUM);
}

Tensor Tensor::min(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id].min(VALUE_TAPE[X.tape_id]), std::pair<int, int>(this->tape_id, X.tape_id), MIN_M);
}

Tensor Tensor::max(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id].max(VALUE_TAPE[X.tape_id]), std::pair<int, int>(this->tape_id, X.tape_id), MAX_M);
}

Tensor Tensor::min(float y)
{
	FLOAT_TAPE[idt] = y;
	return Tensor(VALUE_TAPE[this->tape_id].min(y), std::pair<int, int>(this->tape_id, -1), MIN_N);
}

Tensor Tensor::max(float y)
{
	FLOAT_TAPE[idt] = y;
	return Tensor(VALUE_TAPE[this->tape_id].max(y), std::pair<int, int>(this->tape_id, -1), MAX_N);
}

Tensor Tensor::indicies(int dim)
{
	Tensor C(VALUE_TAPE[this->tape_id].indicies(dim), std::pair<int,int>(this->tape_id,-1), GET_INDEX);
	return C;
}

void Tensor::reshape(int x, int y, int z, int w)
{
	//TODO
}

Tensor Tensor::transpose(int dim_a, int dim_b)
{
	TRANSPOSE_TAPE[idt] = std::pair<int,int>(dim_a,dim_b);
	return Tensor(VALUE_TAPE[this->tape_id].transpose(dim_a,dim_b), std::pair<int, int>(this->tape_id, -1), TRANSPOSE);
}

Tensor Tensor::repeat(int n)
{
	REPEAT_TAPE[idt] = n;
	return Tensor(VALUE_TAPE[this->tape_id].repeat(n), std::pair<int, int>(this->tape_id, -1), REPEAT);
}

Tensor Tensor::_if(Tensor & _true, float _false)
{
	FLOAT_TAPE[idt] = _false;
	return Tensor(VALUE_TAPE[this->tape_id]._if(VALUE_TAPE[_true.tape_id], TensorCL(VALUE_TAPE[this->tape_id], _false)), std::pair<int, int>(this->tape_id, _true.tape_id), IF_COND);
}

Tensor Tensor::dot(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id].dot(VALUE_TAPE[X.tape_id]), std::pair<int, int>(this->tape_id, X.tape_id), DOT);
}

//Vector Jacobian product, aka backpropagation derivatives
std::pair<int,int> VJP(int outgrad_id, int out_id, Tensor::OPERATION op)
{
	int id_a = PARENTS_TAPE[out_id].first;
	int id_b = PARENTS_TAPE[out_id].second;
	//-1 means no derivative
	switch (op)
	{
	case Tensor::NONE:
		return std::pair<int, int>(-1, -1);
	case Tensor::ADD_T:
		return std::pair<int, int>(outgrad_id, outgrad_id);
	case Tensor::SUBS_T:
		return std::pair<int, int>(outgrad_id, (-Tensor(outgrad_id)).ID());
	case Tensor::MUL_T:
		return std::pair<int, int>((Tensor(outgrad_id)*Tensor(id_b)).ID(), (Tensor(outgrad_id)*Tensor(id_a)).ID());
	case Tensor::DIV_T:
		return std::pair<int, int>((Tensor(outgrad_id) / Tensor(id_b)).ID(), (-Tensor(outgrad_id)*Tensor(id_a)*(Tensor(id_b) ^ 2)).ID());
	case Tensor::NEG:
		return std::pair<int, int>((-Tensor(outgrad_id)).ID(), -1);
	case Tensor::ADD_N:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>(outgrad_id, -1);
	case Tensor::SUBS_N:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>(outgrad_id, -1);
	case Tensor::MUL_N:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>((Tensor(outgrad_id) / num).ID(), -1);
	case Tensor::DIV_N:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>((Tensor(outgrad_id) * num).ID(), -1);
	case Tensor::SIN:
		return std::pair<int, int>((cos(Tensor(id_a))*Tensor(outgrad_id)).ID(), -1);
	case Tensor::COS:
		return std::pair<int, int>((-sin(Tensor(id_a))*Tensor(outgrad_id)).ID(), -1);
	case Tensor::TAN:
		return std::pair<int, int>((Tensor(outgrad_id) / (cos(Tensor(id_a)) ^ 2)).ID(), -1);
	case Tensor::LOG:
		return std::pair<int, int>((Tensor(outgrad_id) / (Tensor(id_a))).ID(), -1);
	case Tensor::TANH:
		return std::pair<int, int>((Tensor(outgrad_id) * (1 - Tensor(out_id) ^ 2)).ID(), -1);
	case Tensor::POW:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>((Tensor(outgrad_id) * num * Tensor(id_a) ^ (num - 1)).ID(), -1);
	case Tensor::EXP:
		return std::pair<int, int>((Tensor(outgrad_id) * Tensor(out_id)).ID(), -1);
	case Tensor::SUM:
		int rnk = VALUE_TAPE[id_a].GetParam().rank;
		int N = VALUE_TAPE[id_a].GetParam().size[rnk - 1];
		return std::pair<int, int>((repeat(Tensor(outgrad_id), N).ID()), -1);
	case Tensor::MIN_M:
		return std::pair<int, int>(_if(Tensor(id_a) < Tensor(id_b), Tensor(outgrad_id), 0.f).ID(), _if(Tensor(id_a) > Tensor(id_b), Tensor(outgrad_id), 0.f).ID());
	case Tensor::MAX_M:
		return std::pair<int, int>(_if(Tensor(id_a) > Tensor(id_b), Tensor(outgrad_id), 0.f).ID(), _if(Tensor(id_a) < Tensor(id_b), Tensor(outgrad_id), 0.f).ID());
	case Tensor::MIN_N:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>(_if(Tensor(id_a) < num, Tensor(outgrad_id), 0.f).ID(), _if(Tensor(id_a) > num, Tensor(outgrad_id), 0.f).ID());
	case Tensor::MAX_N:
		float num = FLOAT_TAPE[out_id];
		return std::pair<int, int>(_if(Tensor(id_a) > num, Tensor(outgrad_id), 0.f).ID(), _if(Tensor(id_a) < num, Tensor(outgrad_id), 0.f).ID());
	case Tensor::TRANSPOSE:
		int dim_a = TRANSPOSE_TAPE[out_id].first;
		int dim_b = TRANSPOSE_TAPE[out_id].second;
		return std::pair<int, int>(transpose(Tensor(outgrad_id), dim_a, dim_b).ID(), -1);
	case Tensor::DOT:
		int rnk = VALUE_TAPE[id_b].GetParam().rank - 2;
		int DA = dot(Tensor(outgrad_id), transpose(Tensor(id_b))).ID();
		int DB = dot(transpose(Tensor(id_a)), Tensor(outgrad_id)).ID();
		for (int i = 0; i < rnk; i++) //sum over all redundant dimensions
		{
			DA = sum(Tensor(DA)).ID();
		}
		return std::pair<int, int>(DA, DB);
	case Tensor::REPEAT:
		return std::pair<int, int>(sum(Tensor(outgrad_id)).ID(), -1);
	case Tensor::IF_COND:
		return std::pair<int, int>(-1, _if(Tensor(id_a), Tensor(outgrad_id), 0.f).ID());
	default:
		return std::pair<int, int>(-1, -1);
	}
}

Tensor Tensor::Derivative_WRT(Tensor & wrt)
{
	
}

TensorCL& Tensor::GetTensor()
{
	return VALUE_TAPE[tape_id];
}

int Tensor::ID()
{
	return tape_id;
}

//it's slow, but whatever
std::vector<int> Tensor::FindChilds(int id)
{
	std::vector<int> childs;
	auto it = PARENTS_TAPE.begin();
	// Iterate through the map
	while (it != PARENTS_TAPE.end())
	{
		// Check if parent id of this entry matches with given id
		if (it->second.first == id || it->second.second == id)
		{
			childs.push_back(it->first);
		}
		// Go to next entry in map
		it++;
	}
	return childs;
}

void Tensor::RecursiveDestruction(int id)
{
	if (VALUE_TAPE.count(id) != 0) //if not yet deleted
	{
		std::vector<int> childs = FindChilds(id);

		for (int i = 0; i < childs.size(); i++)
		{
			RecursiveDestruction(childs[i]);
		}

		VALUE_TAPE.erase(id);
		OPERATION_TAPE.erase(id);
		PARENTS_TAPE.erase(id);
	}
}

Tensor::~Tensor()
{
	if (VALUE_TAPE.count(tape_id) != 0) //if not yet deleted
	{
		//if this node is of operation type none -> then delete the entire tree since it can't be used out of scope anyway
		if (OPERATION_TAPE[tape_id] == NONE)
		{
			RecursiveDestruction(tape_id);
		}

		//delete everything from previous states
		for (int i = 0; i < old_ids.size(); i++)
		{
			if (OPERATION_TAPE[old_ids[i]] == NONE)
			{
				RecursiveDestruction(old_ids[i]);
			}
		}
	}
}

Tensor::Tensor(Tensor & X)
{
	tape_id = X.tape_id;
}

Tensor::Tensor(Tensor && X)
{
	tape_id = X.tape_id;
	X.tape_id = -100;
	old_ids = X.old_ids;
	X.old_ids.clear();
}

Tensor & Tensor::operator=(Tensor & X)
{
	tape_id = X.tape_id;
	return *this;
}

Tensor & Tensor::operator=(Tensor && X)
{
	tape_id = X.tape_id;
	X.tape_id = -100;
	old_ids = X.old_ids;
	X.old_ids.clear();
	return *this;
}

Tensor & Tensor::operator=(float a)
{
	TensorCL C(VALUE_TAPE[tape_id].GetParam());
	C = a;
	old_ids.push_back(tape_id);
	init(C);
	return *this;
}

Tensor Tensor::operator+(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id] + VALUE_TAPE[X.tape_id], std::pair<int,int>(this->tape_id, X.tape_id), ADD_T);
}

Tensor Tensor::operator-(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id] - VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), SUBS_T);
}

Tensor Tensor::operator*(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id] * VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), MUL_T);
}

Tensor Tensor::operator/(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id] / VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), DIV_T);
}

Tensor Tensor::operator+(float x)
{
	FLOAT_TAPE[idt] = x;
	return Tensor(VALUE_TAPE[this->tape_id] + x, std::pair<int, int>(this->tape_id, -1), ADD_N);
}

Tensor Tensor::operator-(float x)
{
	FLOAT_TAPE[idt] = x;
	return Tensor(VALUE_TAPE[this->tape_id] - x, std::pair<int, int>(this->tape_id, -1), SUBS_N);
}

Tensor Tensor::operator-()
{
	return Tensor(-VALUE_TAPE[this->tape_id], std::pair<int, int>(this->tape_id, -1), NEG);
}

Tensor Tensor::operator*(float x)
{
	FLOAT_TAPE[idt] = x;
	return Tensor(VALUE_TAPE[this->tape_id] * x, std::pair<int, int>(this->tape_id, -1), MUL_N);
}

Tensor Tensor::operator/(float x)
{
	FLOAT_TAPE[idt] = x;
	return Tensor(VALUE_TAPE[this->tape_id] / x, std::pair<int, int>(this->tape_id, -1), DIV_N);
}

Tensor Tensor::operator>(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id] > VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), MORE_M);
}

Tensor Tensor::operator<(Tensor & X)
{
	return Tensor(VALUE_TAPE[this->tape_id] < VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), LESS_M);
}

Tensor Tensor::operator>(float x)
{
	FLOAT_TAPE[idt] = x;
	return Tensor(VALUE_TAPE[this->tape_id] > x, std::pair<int, int>(this->tape_id, -1), MORE_N);
}

Tensor Tensor::operator<(float x)
{
	FLOAT_TAPE[idt] = x;
	return Tensor(VALUE_TAPE[this->tape_id] < x, std::pair<int, int>(this->tape_id, -1), LESS_N);
}

std::string getOperationName(Tensor::OPERATION op)
{
	switch (op)
	{
	case Tensor::NONE:
		return "NONE";
	case Tensor::ADD_T:
		return "ADD_TENSORS";
	case Tensor::SUBS_T:
		return "SUBSTRACT_TENSORS";
	case Tensor::MUL_T:
		return "MULTIPLY_TENSORS";
	case Tensor::DIV_T:
		return "DIVIDE_TENSORS";
	case Tensor::NEG:
		return "NEGATE";
	case Tensor::ADD_N:
		return "ADD_NUMBER";
	case Tensor::SUBS_N:
		return "SUBSTRACT_NUMBER";
	case Tensor::MUL_N:
		return "MULTIPLY_NUMBER";
	case Tensor::DIV_N:
		return "DIVIDE_NUMBER";
	case Tensor::SIN:
		return "SIN";
	case Tensor::COS:
		return "COS";
	case Tensor::TAN:
		return "EXP";
	case Tensor::LOG:
		return "LOG";
	case Tensor::TANH:
		return "TANH";
	case Tensor::POW:
		return "POW";
	case Tensor::SUM:
		return "SUM";
	case Tensor::MIN_M:
		return "MIN_M";
	case Tensor::MAX_M:
		return "MAX_M";
	case Tensor::MIN_N:
		return "MIN_N";
	case Tensor::MAX_N:
		return "MAX_N";
	case Tensor::TRANSPOSE:
		return "TRANSPOSE";
	case Tensor::DOT:
		return "DOT";
	case Tensor::REPEAT:
		return "REPEAT";
	default:
		return "UNKNOWN";
	}
}

void PrintTAPE(bool disp_value)
{
	std::cout << "TAPE:" << std::endl;
	auto it = PARENTS_TAPE.begin();
	auto it2 = OPERATION_TAPE.begin();
	auto it3 = VALUE_TAPE.begin();
	auto i = 0;
	// Iterate through the map
	while (it != PARENTS_TAPE.end())
	{
		std::cout <<"Record_" <<i << " :: " << it->second.first << " * " << it->second.second << " " << getOperationName(it2->second) << std::endl;
		if (disp_value)
		{
			PrintTensor(it3->second);
		}
		it++; it2++; it3++;
		i++;
	}
}

void PrintTensor(Tensor & a)
{
	PrintTensor(a.GetTensor());
}

Tensor operator+(float x, Tensor & Y)
{
	return Y + x;
}

Tensor operator-(float x, Tensor & Y)
{
	return -Y + x;
}

Tensor operator*(float x, Tensor & Y)
{
	return Y * x;
}

Tensor operator/(float x, Tensor & Y)
{
	return Y^(-1.f) * x;
}

Tensor operator>(float x, Tensor & Y)
{
	return Y < x;
}

Tensor operator<(float x, Tensor & Y)
{
	return Y > x;
}

Tensor sin(Tensor & X)
{
	return X.sin();
}

Tensor cos(Tensor & X)
{
	return X.cos();
}

Tensor tan(Tensor & X)
{
	return X.tan();
}

Tensor exp(Tensor & X)
{
	return X.exp();
}

Tensor log(Tensor & X)
{
	return X.log();
}


Tensor tanh(Tensor & X)
{
	return X.tanh();
}

Tensor sum(Tensor & X)
{
	return X.sum();
}

Tensor min(Tensor & X, Tensor & Y)
{
	return X.min(Y);
}

Tensor max(Tensor & X, Tensor & Y)
{
	return X.max(Y);
}

Tensor min(Tensor & X, float y)
{
	return X.min(y);
}

Tensor max(Tensor & X, float y)
{
	return X.max(y);
}

Tensor min(float y, Tensor & X)
{
	return X.max(y);
}

Tensor max(float y, Tensor & X)
{
	return X.min(y);
}

Tensor dot(Tensor & X, Tensor & Y)
{
	return X.dot(Y);
}

Tensor indicies(Tensor & X, int dim)
{
	return  X.indicies(dim);
}

Tensor repeat(Tensor & X, int n)
{
	return X.repeat(n);
}

Tensor transpose(Tensor & X, int dim_a, int dim_b)
{
	return X.transpose(dim_a, dim_b);
}

Tensor _if(Tensor & _cond, Tensor & _true, float _false)
{
	return _cond._if(_true,_false);
}

Gradient::Gradient(Tensor END)
{
}

Gradient::Gradient(Gradient & A)
{
	dydx = A.dydx;
}

Gradient & Gradient::operator=(Gradient & X)
{
	dydx = X.dydx;
	return *this;
}

Tensor Gradient::wrt(Tensor & X)
{
	if (VALUE_TAPE.count(dydx[X.ID()]) != 0) //if exists
	{
		return VALUE_TAPE[X.ID()];
	}
	else
	{
		return X;
	}	
}

void Gradient::ComputeDerivative(int parent_id, int child_id)
{
}
