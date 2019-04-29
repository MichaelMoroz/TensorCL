#include "Optimizer.h"

Optimizer::Optimizer(OPTIMIZATION_METHOD method)
{
	method_used = method;
}

void Optimizer::AddParameter(Tensor & X)
{
	OPTIM_TENSORS.push_back(X.ID());
}

void Optimizer::Optimization_Cost(Tensor & COST)
{
	cost_id = COST.ID();
}

void Optimizer::OptimizationIteration(float dt)
{

}
