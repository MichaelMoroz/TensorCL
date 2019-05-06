#include "Optimizer.h"

Optimizer::Optimizer()
{
	method_used = NONE;
	iterations = 0;
}

Optimizer::Optimizer(OPTIMIZATION_METHOD method)
{
	method_used = method;
	iterations = 0;
}

void Optimizer::setMethod(OPTIMIZATION_METHOD method)
{
	if(OPTIM_TENSORS.size() == 0)
		method_used = method;
}

void Optimizer::AddParameter(Tensor & X)
{
	if (method_used != NONE)
	{
		OPTIM_TENSORS.push_back(X);

		switch (method_used)
		{
		case ADAM:
			moment[X.ID()] = TensorCL(X.GetTensor(), 0.f);
			second_moment[X.ID()] = TensorCL(X.GetTensor(), 0.f);
			break;
		default:
			break;
		}
	}
}

void Optimizer::Optimization_Cost(Tensor & COST)
{
	cost_id = COST.ID();
}

void Optimizer::OptimizationIteration(float dt, bool print_grad)
{
	if (method_used != NONE)
	{
		iterations++;
		std::unique_ptr<Gradient> grad;
		float beta_1 = 0.9f, beta_2 = 0.999f, epsilon = 1e-4f;
		switch (method_used)
		{
		case GRAD_DESC:
			grad.reset(new Gradient(cost_id));
			for (Tensor &tensor : OPTIM_TENSORS)
			{
				//operate only on base tensors, we wont need to find the gradient of gradient descent anyway =)
				tensor.GetTensor() -= grad->wrt(tensor).GetTensor()*dt;
				if (print_grad)
				{
					PrintTensor(grad->wrt(tensor));
				}
			}
			break;
		case ADAM:
			grad.reset(new Gradient(cost_id));
			for (Tensor &tensor : OPTIM_TENSORS)
			{
				moment[tensor.ID()] = moment[tensor.ID()] * beta_1 + grad->wrt(tensor).GetTensor() * (1 - beta_1);
				second_moment[tensor.ID()] = second_moment[tensor.ID()] * beta_2 + (grad->wrt(tensor).GetTensor() ^ 2)*(1 - beta_2);
				TensorCL mhat = moment[tensor.ID()] / (1 - pow(beta_1, iterations));
				TensorCL vhat = second_moment[tensor.ID()] / (1 - pow(beta_2, iterations));
				tensor.GetTensor() -= mhat * ((vhat + epsilon) ^ 0.5f)*dt;
				if (print_grad)
				{
					PrintTensor(grad->wrt(tensor));
				}
			}
			break;
		default:
			break;
		}
	}
	
}

Optimizer::~Optimizer()
{
	OPTIM_TENSORS.clear();
	moment.clear();
	second_moment.clear();
}
