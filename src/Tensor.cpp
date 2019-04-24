#pragma once
#include <Tensor.h>
#include <vector>


Tensor::Tensor(TensorCL input)
{
	tape_id = VALUE_TAPE.size();
	VALUE_TAPE.push_back(input);
	OPERATION_TAPE.push_back(NONE);
	PARENTS_TAPE.push_back(std::pair<int, int>(NULL, NULL));
}

Tensor::~Tensor()
{
	
}

void Tensor::CLEAR_TAPE()
{
	VALUE_TAPE.clear();
	OPERATION_TAPE.clear();
	PARENTS_TAPE.clear();
}
