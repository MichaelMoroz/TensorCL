#pragma once
#include <Tensor.h>
#include <vector>
#include <stack>

int idt = 0;

#define DEBUG false
#define DEBUG_PRINT_VALUE false
#define DEBUG_PRINT_ARGS true

#define DEBUG_VJP false
#define DEBUG_PRINT_TAPE true

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
	case Tensor::IF_COND:
		return "IF_COND";
	case Tensor::DOT:
		return "DOT";
	case Tensor::REPEAT:
		return "REPEAT";
	default:
		return "UNKNOWN";
	}
}

// operation trees/recording tape
// only one instance for the entire program exists
std::map<int, TensorCL> VALUE_TAPE;
std::map<int, Tensor::OPERATION> OPERATION_TAPE;
std::map<int, std::pair<int, int> > PARENTS_TAPE;
std::map<int, std::vector<int> > CHILDS_TAPE;
std::map<int, float> FLOAT_TAPE;
std::map<int, std::pair<int, int> > TRANSPOSE_TAPE;
std::map<int, int> NUMBER_OF_COPIES;

Tensor::Tensor()
{
	tape_id = -1;
}

Tensor::Tensor(Size s, float fill, bool rand)
{
	init(TensorCL(s, fill, rand));
}

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
	if (id >= 0)
	{
		NUMBER_OF_COPIES[tape_id] ++;
	}
}

Tensor::Tensor(Tensor & x, float fill)
{
	*this = Tensor(TensorCL(VALUE_TAPE[x.tape_id], fill), std::pair<int,int>(x.tape_id, -1), NOT);
}

Tensor::Tensor(TensorData & A)
{
	*this = Tensor(TensorCL(A), std::pair<int, int>(-1, -1), NONE);
}

void Tensor::init(TensorCL & X, std::pair<int, int> parents, OPERATION op)
{
	//Add this operation and its value to the tape
	tape_id = idt;
	VALUE_TAPE[idt] = X;
	OPERATION_TAPE[idt] = op;
	PARENTS_TAPE[idt] = parents;
	if (parents.first != -1) CHILDS_TAPE[parents.first].push_back(idt);
	if (parents.second != -1) CHILDS_TAPE[parents.second].push_back(idt);
	NUMBER_OF_COPIES[idt] = 1;

	#if DEBUG
		std::cout << "TapeID_" << idt << " :: " << getOperationName(op) << "(arg1_id = " << parents.first << ", arg2_id = " << parents.second << ") " << std::endl;
		#if DEBUG_PRINT_VALUE
			#if DEBUG_PRINT_ARGS
			if (parents.first != -1)
			{
				std::cout << "arg1_id = " << parents.first << std::endl;
				PrintTensor(VALUE_TAPE[parents.first]);
			}
			if (parents.second != -1)
			{
				std::cout << "arg1_id = " << parents.second << std::endl;
				PrintTensor(VALUE_TAPE[parents.second]);
			}
			#endif
		
			PrintTensor(VALUE_TAPE[idt]);
		#endif

		#if DEBUG_PRINT_TAPE
					PrintTAPE(false);
		#endif
	#endif

	idt++;
}

Tensor Tensor::diag(float x, float y)
{
	return Tensor(VALUE_TAPE[this->tape_id].diag(x,y), std::pair<int, int>(this->tape_id, -1), NOT);
}

Tensor Tensor::random()
{
	return Tensor(VALUE_TAPE[this->tape_id].random(), std::pair<int, int>(this->tape_id, -1), NOT);
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

Tensor Tensor::pow(float y)
{
	if (y == 1) //dont use pow if power is = 1
	{
		return *this;
	}
	else if (y == 0) // when pow = 0 then tensor = 1
	{
		FLOAT_TAPE[idt] = y;
		return Tensor(TensorCL(VALUE_TAPE[this->tape_id],1.f), std::pair<int, int>(this->tape_id, -1), POW);
	}
	else
	{
		FLOAT_TAPE[idt] = y;
		return Tensor(VALUE_TAPE[this->tape_id].pow(y), std::pair<int, int>(this->tape_id, -1), POW);
	}
}

int Tensor::operator[](int dim)
{
	return GetTensor().GetParam().size[dim];
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

TensorCL& Tensor::GetTensor()
{
	return VALUE_TAPE[tape_id];
}

int Tensor::ID()
{
	return tape_id;
}

std::vector<int> Tensor::FindChilds(int id)
{
	//the slow way, do not use!
	/*std::vector<int> childs;
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
	}*/
	return CHILDS_TAPE[id];
}


void RemoveFromTape(int id)
{
	VALUE_TAPE.erase(id);
	OPERATION_TAPE.erase(id);
	PARENTS_TAPE.erase(id);
	NUMBER_OF_COPIES.erase(id);
	CHILDS_TAPE.erase(id);
	FLOAT_TAPE.erase(id);
	TRANSPOSE_TAPE.erase(id);
}

void Tensor::RecursiveDestructionChilds(int id)
{
	if (VALUE_TAPE.count(id) != 0) //if not yet deleted
	{
		for (auto &child_id: FindChilds(id))
		{
			RecursiveDestructionChilds(child_id);
		}

		if (NUMBER_OF_COPIES[id] == 0)
		{
			RemoveFromTape(id);
		}
	}
}

void Tensor::RecursiveDestructionParents(int id)
{
	if (VALUE_TAPE.count(id) != 0) //if not yet deleted
	{
		if (NUMBER_OF_COPIES[id] == 0)
		{
			if (AreAllChildsDestroyed(id))
			{
				RecursiveDestructionParents(PARENTS_TAPE[id].first);
				RecursiveDestructionParents(PARENTS_TAPE[id].second);

				RemoveFromTape(id);
			}
		}
	}
}

bool Tensor::AreAllChildsDestroyed(int id)
{
	bool tree_destroyed = NUMBER_OF_COPIES[id] == 0;

	if (tree_destroyed)
	{
		for (auto &child_id : FindChilds(id))
		{
			tree_destroyed = tree_destroyed && AreAllChildsDestroyed(child_id);
		}
	}
	
	return tree_destroyed;
}

void Tensor::Destroy(int id)
{
	if (VALUE_TAPE.count(id) != 0) //if not yet deleted
	{
		if(NUMBER_OF_COPIES[id] > 0)
			NUMBER_OF_COPIES[id] -= 1;
		if (AreAllChildsDestroyed(id))
		{
			RecursiveDestructionParents(id);
			RecursiveDestructionChilds(id);
		}
	}
}

Tensor::~Tensor()
{
	//destroy everything from previous states
	for (auto &id:old_ids)
	{
		Destroy(id);
	}

	Destroy(tape_id);
}

Tensor::Tensor(const Tensor & X)
{
	tape_id = X.tape_id;
	NUMBER_OF_COPIES[tape_id]++;
}

Tensor::Tensor(Tensor && X)
{
	tape_id = X.tape_id;
	X.tape_id = -100;
	old_ids = X.old_ids;
	X.old_ids.clear();
}

Tensor & Tensor::operator=(const Tensor & X)
{
	old_ids.push_back(tape_id);
	tape_id = X.tape_id;
	NUMBER_OF_COPIES[tape_id]++;
	return *this;
}

Tensor & Tensor::operator=(Tensor && X)
{
	old_ids.push_back(tape_id);
	tape_id = X.tape_id;
	X.tape_id = -100;
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
	if (X.ID() != -1 && ID() != -1)
	{
		return Tensor(VALUE_TAPE[this->tape_id] + VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), ADD_T);
	}
	else if (ID() == -1)
	{
		return X;
	}
	else
	{
		return *this;
	}
}

Tensor Tensor::operator-(Tensor & X)
{
	if (X.ID() != -1 && ID() != -1)
	{
		return Tensor(VALUE_TAPE[this->tape_id] - VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), SUBS_T);
	}
	else if (ID() == -1 && ID() != -1)
	{
		return X;
	}
	else
	{
		return *this;
	}
}

Tensor Tensor::operator*(Tensor & X)
{
	if (X.ID() != -1 && ID() != -1)
	{
		return Tensor(VALUE_TAPE[this->tape_id] * VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), MUL_T);
	}
	else if (ID() == -1)
	{
		return X;
	}
	else
	{
		return *this;
	}
}

Tensor Tensor::operator/(Tensor & X)
{
	if (X.ID() != -1 && ID() != -1)
	{
		return Tensor(VALUE_TAPE[this->tape_id] / VALUE_TAPE[X.tape_id], std::pair<int, int>(this->tape_id, X.tape_id), DIV_T);
	}
	else if (ID() == -1)
	{
		return X;
	}
	else
	{
		return *this;
	}
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

float Tensor::operator()(int i, int j, int k, int m)
{
	return GetTensor()(i, j, k, m);
}

int TAPE_SIZE()
{
	return VALUE_TAPE.size();
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
		std::cout <<"Record_" << it->first << " :: " << getOperationName(it2->second) << "(arg1_id = " << it->second.first << ", arg2_id = " << it->second.second << ") " << std::endl;
		if (disp_value)
		{
			PrintTensor(it3->second);
		}
		it++; it2++; it3++;
		i++;
	}
}

bool exists(int id)
{
	if (VALUE_TAPE.count(id) != 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool hasvisited(int id, std::map<int, bool> &visited)
{
	if (visited.count(id) != 0) //if exists
	{
		return true;
	}
	else
	{
		return false;
	}
}

void toposort_recursive(int node, std::vector<int> &Stack, std::map<int, bool> &visited)
{
	visited[node] = true;

	if (PARENTS_TAPE[node].first != -1 
		&& !hasvisited(PARENTS_TAPE[node].first, visited) 
		&& exists(PARENTS_TAPE[node].first))
	{
		toposort_recursive(PARENTS_TAPE[node].first, Stack, visited);
	}

	if (PARENTS_TAPE[node].second != -1 
		&& !hasvisited(PARENTS_TAPE[node].second, visited) 
		&& exists(PARENTS_TAPE[node].second))
	{
		toposort_recursive(PARENTS_TAPE[node].second, Stack, visited);
	}

	Stack.push_back(node);
}

std::vector<int> toposort(int end_node)
{
	std::vector<int> Stack;
	std::map<int, bool> visited;
	toposort_recursive(end_node, Stack, visited);

	std::vector<int> sorted;
	while (Stack.empty() == false)
	{
		sorted.push_back(Stack.back());
		Stack.pop_back();
	}
	return sorted;
}

//Vector Jacobian product, aka backpropagation derivatives
void Gradient::VJP(int outgrad_id, int out_id, Tensor::OPERATION op)
{
	int id_a = PARENTS_TAPE[out_id].first;
	int id_b = PARENTS_TAPE[out_id].second;
	float num;
	int dim_a, dim_b, rnk, N, DA, DB;
	Tensor P1(id_a), P2(id_b);

#if DEBUG_VJP
	std::cout << "GradID: " << outgrad_id << ", OutID: " << out_id << ", Operation: " <<getOperationName(op) << "(arg1_id = " << id_a << ", arg2_id = " << id_b <<")"<< std::endl;
	
	#if DEBUG_PRINT_VALUE
		PrintTensor(VALUE_TAPE[outgrad_id]);

		PrintTensor(VALUE_TAPE[out_id]);

		if (id_a != -1)
		{
			PrintTensor(VALUE_TAPE[id_a]);
		}

		if (id_b != -1)
		{
			PrintTensor(VALUE_TAPE[id_b]);
		}
	#endif

	#if DEBUG_PRINT_TAPE
		PrintTAPE(false);
	#endif

#endif
	
	//-1 means no derivative
	switch (op)
	{
	case Tensor::ADD_T:
		AddDerivative(id_a, Tensor(outgrad_id));
		AddDerivative(id_b, Tensor(outgrad_id));
		break;
	case Tensor::SUBS_T:
		AddDerivative(id_a, Tensor(outgrad_id));
		AddDerivative(id_b, -Tensor(outgrad_id));
		break;
	case Tensor::MUL_T:
		AddDerivative(id_a, Tensor(outgrad_id)*Tensor(id_b));
		AddDerivative(id_b, Tensor(outgrad_id)*Tensor(id_a));
		break;
	case Tensor::DIV_T:
		AddDerivative(id_a, Tensor(outgrad_id) / Tensor(id_b));
		AddDerivative(id_b, -Tensor(outgrad_id)*Tensor(id_a)*pow(Tensor(id_b),-2.f));
		break;
	case Tensor::NEG:
		AddDerivative(id_a, -Tensor(outgrad_id));
		break;
	case Tensor::ADD_N:
	    num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, Tensor(outgrad_id));
		break;
	case Tensor::SUBS_N:
		num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, Tensor(outgrad_id));
		break;
	case Tensor::MUL_N:
		num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, Tensor(outgrad_id) * num);
		break;
	case Tensor::DIV_N:
		num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, Tensor(outgrad_id) / num);
		break;
	case Tensor::SIN:
		AddDerivative(id_a, cos(Tensor(id_a))*Tensor(outgrad_id));
		break;
	case Tensor::COS:
		AddDerivative(id_a, -sin(Tensor(id_a))*Tensor(outgrad_id));
		break;
	case Tensor::TAN:
		AddDerivative(id_a, Tensor(outgrad_id) * pow(cos(Tensor(id_a)), -2.f) );
		break;
	case Tensor::LOG:
		AddDerivative(id_a, Tensor(outgrad_id) / Tensor(id_a));
		break;
	case Tensor::TANH:
		AddDerivative(id_a, Tensor(outgrad_id) * (1 - pow(Tensor(out_id), 2)));
		break;
	case Tensor::POW:
		num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, Tensor(outgrad_id) * pow(Tensor(id_a), num - 1.f) * num);
		break;
	case Tensor::EXP:
		AddDerivative(id_a, Tensor(outgrad_id) * Tensor(out_id));
		break;
	case Tensor::SUM:
		rnk = VALUE_TAPE[id_a].GetParam().rank;
		N = VALUE_TAPE[id_a].GetParam().size[rnk - 1];
		AddDerivative(id_a, repeat(Tensor(outgrad_id), N));
		break;
	case Tensor::MIN_M:
		AddDerivative(id_a, _if(Tensor(id_a) < Tensor(id_b), Tensor(outgrad_id), 0.f));
		AddDerivative(id_b, _if(Tensor(id_a) > Tensor(id_b), Tensor(outgrad_id), 0.f));
		break;
	case Tensor::MAX_M:
		AddDerivative(id_a, _if(Tensor(id_a) > Tensor(id_b), Tensor(outgrad_id), 0.f));
		AddDerivative(id_b, _if(Tensor(id_a) < Tensor(id_b), Tensor(outgrad_id), 0.f));
		break;
	case Tensor::MIN_N:
		num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, _if(Tensor(id_a) < num, Tensor(outgrad_id), 0.f));
		break;
	case Tensor::MAX_N:
		num = FLOAT_TAPE[out_id];
		AddDerivative(id_a, _if(Tensor(id_a) > num, Tensor(outgrad_id), 0.f));
		break;
	case Tensor::TRANSPOSE:
		dim_a = TRANSPOSE_TAPE[out_id].first;
		dim_b = TRANSPOSE_TAPE[out_id].second;
		AddDerivative(id_a, transpose(Tensor(outgrad_id), dim_a, dim_b));
		break;
	case Tensor::DOT:
		rnk = getrank(VALUE_TAPE[id_b].GetParam())- 2;
		if (getrank(VALUE_TAPE[outgrad_id].GetParam()) == 1)
		{
			P1 = dot(transpose(Tensor(outgrad_id)), transpose(Tensor(id_b)));
			P2 = dot(transpose(Tensor(id_a)), transpose(Tensor(outgrad_id)));
		}
		else
		{
			P1 = dot(Tensor(outgrad_id), transpose(Tensor(id_b)));
			P2 = dot(transpose(Tensor(id_a)), Tensor(outgrad_id));
		}
		
		for (int i = 0; i < rnk; i++) //sum over all redundant dimensions
		{
			P1 = sum(P1);
		}
		AddDerivative(id_a, P1);
		AddDerivative(id_b, P2);
		break;
	case Tensor::REPEAT:
		rnk = VALUE_TAPE[outgrad_id].GetParam().rank;
		N = VALUE_TAPE[outgrad_id].GetParam().size[rnk - 1];
		AddDerivative(id_a, sum(Tensor(outgrad_id)));
		break;
	case Tensor::IF_COND:
		AddDerivative(id_b, _if(Tensor(id_a), Tensor(outgrad_id), 0.f));
		break;
	default:
		break;
	}
}

Gradient::Gradient(Tensor END)
{
	Tensor outgrad(END, 1.f); //initial gradient
	dydx[END.ID()] = outgrad;
	#if DEBUG_VJP
		PrintTAPE(false);
	#endif
	std::vector<int> toposorted_nodes = toposort(END.ID());
	#if DEBUG_VJP
		PrintTAPE(false);
	#endif
	for (auto &node_id : toposorted_nodes)
	{
		int grad_id = dydx[node_id].ID();
		if (OPERATION_TAPE[node_id] != Tensor::NONE)
		{
			VJP(grad_id, node_id, OPERATION_TAPE[node_id]);
		}
	}
}

Gradient::Gradient(int tensor_id)
{
	*this = Gradient(Tensor(tensor_id));
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

Tensor& Gradient::wrt(Tensor & X)
{
	return wrt(X.ID());
}

Tensor& Gradient::wrt(int tensor_id)
{
	if (dydx.count(tensor_id) != 0) //if exists
	{
		return dydx[tensor_id];
	}
	else
	{
		return Tensor(-1);
	}
}

Gradient::~Gradient()
{
	dydx.clear();
}


void Gradient::AddDerivative(int pnode, Tensor gnode)
{
	if (pnode != -1)
	{
		if (dydx.count(pnode) != 0) //if a derivative already exists exists
		{
			dydx[pnode] = dydx[pnode] + gnode;
		}
		else
		{
			dydx[pnode] = gnode;
		}
	}
}
