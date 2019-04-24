#include <CL_TENSOR.h>


//TensorCL wrapper with automatic differentiation
class Tensor
{
public:
	Tensor(TensorCL input);
	~Tensor();

	void CLEAR_TAPE();

private:

	enum OPERATION
	{
		NONE,
		ADD_T, SUBS_T, MUL_T, DIV_T, NEG,
		ADD_N, SUBS_N, MUL_N, DIV_N,
		SIN, COS, TAN, EXP, LOG, TANH, POW,
		SUM, MIN_M, MAX_M, MIN_N, MAX_N,
		TRANSPOSE, DOT
	};

	//the id of this element inside the tape
	int tape_id;

	// operation trees/recording tape
	// only one instance exists
	static std::vector<TensorCL> VALUE_TAPE;
	static std::vector<OPERATION> OPERATION_TAPE;
	static std::vector< std::pair<int, int> > PARENTS_TAPE;
};

