#pragma once
#include <Tensor.h>
#include <vector>

int idt = 0;
// operation trees/recording tape
// only one instance exists
std::map<int, TensorCL> VALUE_TAPE;
std::map<int, Tensor::OPERATION> OPERATION_TAPE;
std::map<int, std::pair<int, int> > PARENTS_TAPE;

Tensor::Tensor(int x, int y, int z, int w)
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

void Tensor::init(TensorCL & X, std::pair<int, int> parents, OPERATION op)
{
	tape_id = idt;
	VALUE_TAPE[idt] = X;
	OPERATION_TAPE.emplace(idt, op);
	PARENTS_TAPE.emplace(idt, parents);
	idt++;
}

TensorCL& Tensor::GetTensor()
{
	return VALUE_TAPE[tape_id];
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

Tensor & Tensor::operator=(Tensor & X)
{
	tape_id = X.tape_id;
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
	return Tensor(VALUE_TAPE[this->tape_id] + x, std::pair<int, int>(this->tape_id, -1), ADD_N);
}

Tensor Tensor::operator-(float x)
{
	return Tensor(VALUE_TAPE[this->tape_id] - x, std::pair<int, int>(this->tape_id, -1), SUBS_N);
}

Tensor Tensor::operator-()
{
	return Tensor(-VALUE_TAPE[this->tape_id], std::pair<int, int>(this->tape_id, -1), NEG);
}

Tensor Tensor::operator*(float x)
{
	return Tensor(VALUE_TAPE[this->tape_id] * x, std::pair<int, int>(this->tape_id, -1), MUL_N);
}

Tensor Tensor::operator/(float x)
{
	return Tensor(VALUE_TAPE[this->tape_id] / x, std::pair<int, int>(this->tape_id, -1), DIV_N);
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
	default:
		return "UNKNOWN";
	}
}

void PrintTAPE()
{
	std::cout << "TAPE:" << std::endl;
	auto it = PARENTS_TAPE.begin();
	auto it2 = OPERATION_TAPE.begin();
	auto i = 0;
	// Iterate through the map
	while (it != PARENTS_TAPE.end())
	{
		std::cout <<"Record_" <<i << " :: " << it->second.first << " * " << it->second.second << " " << getOperationName(it2->second) << std::endl;
		it++; it2++;
		i++;
	}
}
