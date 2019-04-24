#include <CL_TENSOR.h>


//TensorCL wrapper with automatic differentiation
class Tensor
{
public:
	Tensor(TensorCL input);
	~Tensor();

private:
	TensorCL DATA;

	enum OPERATION
	{
		NONE,
		ADD_T, SUBS_T, MUL_T, DIV_T, NEG,
		ADD_N, SUBS_N, MUL_N, DIV_N,
		SIN, COS, TAN, EXP, LOG, TANH, POW,
		SUM, MIN_M, MAX_M, MIN_N, MAX_N,
		TRANSPOSE, DOT
	};

	//the autodiff stuff
	OPERATION op;

	//the id of this element inside the tape
	int tape_id;

	//the id's of the parent nodes
	int node_1, node_2;

};

