#include "Optimizer.h"

#define DEBUG_OPTIM false
#define DEBUG_PRINT_MOMENTS true

Optimizer::Optimizer()
{
	method_used = NONE;
	regul_used = NO;
	iterations = 0;
	epsilon = 1e-4f;
}

Optimizer::Optimizer(OPTIMIZATION_METHOD method)
{
	method_used = method;
	regul_used = NO;
	iterations = 0;
	beta_1 = 0.9f;
	beta_2 = 0.999f;
	epsilon = 1e-4f;
	dt = 0.001;
}

void Optimizer::setSpeed(float speed)
{
	dt = speed;
	beta_1 = 0.9f;
	beta_2 = 0.999f;
	epsilon = 1e-4f;
}

void Optimizer::setMethod(OPTIMIZATION_METHOD method)
{
	if(OPTIM_TENSORS.size() == 0)
		method_used = method;
}

void Optimizer::setRegularization(REGULARIZATION method, float rk)
{
	regul_used = method;
	lambda = rk;
}

void Optimizer::AddParameter(Tensor & X)
{
	if (method_used != NONE)
	{
		OPTIM_TENSORS.push_back(X);

		switch (method_used)
		{
		case ADAM:
			moment[X.ID()] = TensorCL(X.GetSize(), 0.f);
			second_moment[X.ID()] = TensorCL(X.GetSize(), 0.f);
			break;
		default:
			break;
		}
	}
}

void Optimizer::Optimize_Cost(Tensor & COST)
{
	if (method_used != NONE)
	{
		cost_id = COST.ID();
		iterations++;
		std::unique_ptr<Gradient> grad;
		
		switch (method_used)
		{
		case GRAD_DESC:
			grad.reset(new Gradient(cost_id));
			for (Tensor &tensor : OPTIM_TENSORS)
			{
				if (grad->wrt(tensor).ID() != -1)
				{
					//operate only on base tensors, we wont need to find the gradient of gradient descent anyway =)
					tensor.GetTensor() -= grad->wrt(tensor).GetTensor()*dt;
					#if (DEBUG_OPTIM)
						PrintTensor(grad->wrt(tensor));
					#endif
				}
			}
			break;
		case ADAM:
			grad.reset(new Gradient(cost_id));
			for (Tensor &tensor : OPTIM_TENSORS)
			{
				if (grad->wrt(tensor).ID() != -1)
				{
					if (iterations == 1)
					{
						moment[tensor.ID()] = grad->wrt(tensor).GetTensor();
						second_moment[tensor.ID()] = pow(grad->wrt(tensor).GetTensor(), 2.f);
					}
					else
					{
						moment[tensor.ID()] = (moment[tensor.ID()] * beta_1) + (grad->wrt(tensor).GetTensor()  * (1.f - beta_1));
						second_moment[tensor.ID()] = (second_moment[tensor.ID()] * beta_2) + (pow(grad->wrt(tensor).GetTensor(), 2.f)*(1.f - beta_2));
					}

					tensor.GetTensor() -= moment[tensor.ID()] * pow(second_moment[tensor.ID()] + epsilon, -0.5f)*dt;
					#if (DEBUG_OPTIM)
						PrintTensor(grad->wrt(tensor));
						if (DEBUG_PRINT_MOMENTS)
						{
							PrintTensor(moment[tensor.ID()]);
							PrintTensor(second_moment[tensor.ID()]);
						}
					#endif
				}
			}
			break;
		default:
			break;
		}

		switch (regul_used)
		{
		case L2:
			for (Tensor &tensor : OPTIM_TENSORS)
			{
				tensor.GetTensor() -= tensor.GetTensor()*lambda;
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

void Optimizer::Clear()
{
	OPTIM_TENSORS.clear();
	moment.clear();
	second_moment.clear();
}