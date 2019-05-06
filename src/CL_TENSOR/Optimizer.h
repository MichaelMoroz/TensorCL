#pragma once
#include <Tensor.h>

//Optimizer for neural nets and stuff
class Optimizer
{
public:
	enum OPTIMIZATION_METHOD
	{
		NONE,
		GRAD_DESC, RMSprop, MOMENTUM, ADAM, //Gradient based
		LEVENBERG_MARQUARDT, //Second order, slow and TODO
		ES, CMA_ES, //Non grad based, slow, requires recomputing the operation tree a lot
		ESGRAD //hybrid method
	};

	Optimizer();
	Optimizer(OPTIMIZATION_METHOD method);

	void setMethod(OPTIMIZATION_METHOD method);
	void AddParameter(Tensor &X);
	void Optimization_Cost(Tensor &COST);
	void OptimizationIteration(float dt, bool print_grad = false);

	~Optimizer();

protected:
	OPTIMIZATION_METHOD method_used;
	int cost_id;
	int iterations;
	std::vector<Tensor> OPTIM_TENSORS;
	std::map<int, TensorCL> moment, second_moment;
};