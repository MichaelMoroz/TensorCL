#include "CL_TENSOR.h"

OpenCL *CL;
CLFunction add, mul, mad, m_dot;
CLFunction more_m, less_m;
CLFunction more_n, less_n, if_cond;
CLFunction idx, sinfun, cosfun, tanfun, tanhfun, powfun, maxfun, minfun, maxfun_f, minfun_f;
CLFunction expfun, logfun, transposefun, repeatfun;
CLFunction sumfun;

void TensorCL::release()
{
	if(host_data != NULL)
		delete[] host_data;

	if (data != NULL)
		clReleaseMemObject(data);
}

TensorCL::~TensorCL()
{
	release();
}

void TensorCL::init_data(float value)
{
	param.length = 1;
	for (int i = 0; i < param.rank; i++)
	{
		param.length *= param.size[i];
	}
	cl_int status;
	data = clCreateBuffer(CL->default_context(), CL_MEM_READ_WRITE, param.length * sizeof(float), 0, &status);

	if (status != CL_SUCCESS)
	{
		string err = "OpenCL error: " + getOpenCLError(status);
		ERROR_MSG(err.c_str());
	}
	else
	{
		clEnqueueFillBuffer(CL->queue(), data, &value, sizeof(value), 0, param.length * sizeof(float), 0, 0, 0);
	}

	host_data = new float[param.length];
}

TensorCL::TensorCL(int r, vector<int> s)
{
	param.rank = r;
	std::copy(s.begin(), s.end(), param.size);
	init_data();
}


TensorCL::TensorCL(int x, int y, int z, int w) 
{
	param.size[0] = x;
	param.size[1] = y;
	param.size[2] = z;
	param.size[3] = w;

	param.rank = 1;

	if (y > 1)
	{
		param.rank = 2;
	}

	if (z > 1)
	{
		param.rank = 3;
	}

	if (w > 1)
	{
		param.rank = 4;
	}
	init_data();
}

TensorCL::TensorCL(cl_tensor p): param(p)
{
	init_data();
}

TensorCL& TensorCL::operator=(float a)
{
 	*this = MAD(0, a);
	return *this;
}

//r-value -> move 
TensorCL::TensorCL(TensorCL && p): data(p.data), host_data(p.host_data), param(p.param)
{
	p.data = NULL;
	p.host_data = NULL;
}

TensorCL & TensorCL::operator=(TensorCL && p)
{
	std::swap(this->data, p.data);
	std::swap(this->host_data, p.host_data);
	std::swap(this->param, p.param);
	return *this;
}

//l-value -> copy
TensorCL& TensorCL::operator=(TensorCL & X)
{
	param = X.param;
	init_data();
	clEnqueueCopyBuffer(CL->queue(), X.data, data, 0, 0, param.length * sizeof(float), NULL, NULL, NULL);
	CL->queue.flush();
	return *this;
}

TensorCL::TensorCL(TensorCL &X)
{
	param = X.param;
	//if array is temporary - then just copy the CL pointer
	init_data();
	clEnqueueCopyBuffer(CL->queue(), data, X.data, 0, 0, param.length * sizeof(float), NULL, NULL, NULL);
	CL->queue.flush();
}

TensorCL& TensorCL::operator+=(TensorCL &X)
{
	*this = *this + X;
}

TensorCL& TensorCL::operator-=(TensorCL &X)
{
	*this = *this - X;
}

TensorCL& TensorCL::operator*=(TensorCL &X)
{
	*this = *this * X;
}

TensorCL& TensorCL::operator/=(TensorCL &X)
{
	*this = *this / X;
}

TensorCL::TensorCL(TensorCL &X, float fill)
{
	param = X.param;
	init_data(fill);
}

TensorCL TensorCL::operator+(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param); //create a temporary array
		add.SetRange(CL->group_size[0], 1, param.length, 1);
		add.SetArg(0, C.data); //result
		add.SetArg(1, data);
		add.SetArg(2, X.data);
		add.SetArg(3, param);
		add.SetArg(4, 1.f);
		add.SetArg(5, 1.f);
		add.RFlush();
		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::operator-(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param); //create a temporary array
		add.SetRange(CL->group_size[0], 1, param.length, 1);
		add.SetArg(0, C.data); //result
		add.SetArg(1, data);
		add.SetArg(2, X.data);
		add.SetArg(3, param);
		add.SetArg(4, 1.f);
		add.SetArg(5, -1.f);
		add.RFlush();

		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::operator*(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param); //create a temporary array
		mul.SetRange(CL->group_size[0], 1, param.length, 1);
		mul.SetArg(0, C.data); //result
		mul.SetArg(1, data);
		mul.SetArg(2, X.data);
		mul.SetArg(3, param);
		mul.SetArg(4, 1.f);
		mul.SetArg(5, 1.f);
		mul.RFlush();

		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::operator/(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param); //create a temporary array
		mul.SetRange(CL->group_size[0], 1, param.length, 1);
		mul.SetArg(0, C.data); //result
		mul.SetArg(1, data);
		mul.SetArg(2, X.data);
		mul.SetArg(3, param);
		mul.SetArg(4, 1.f);
		mul.SetArg(5, -1.f);
		mul.RFlush();

		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::operator>(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param); //create a temporary array
		more_m.SetRange(CL->group_size[0], 1, param.length, 1);
		more_m.SetArg(0, C.data); //result
		more_m.SetArg(1, data);
		more_m.SetArg(2, X.data);
		more_m.SetArg(3, param);
		more_m.RFlush();
		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::operator<(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param); //create a temporary array
		less_m.SetRange(CL->group_size[0], 1, param.length, 1);
		less_m.SetArg(0, C.data); //result
		less_m.SetArg(1, data);
		less_m.SetArg(2, X.data);
		less_m.SetArg(3, param);
		less_m.RFlush();
		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::operator+(float x)
{
	return MAD(1.f, x);
}

TensorCL TensorCL::operator-(float x)
{
	return MAD(1.f, -x);
}

TensorCL TensorCL::operator-()
{
	return MAD(-1.f, 0);
}

TensorCL TensorCL::operator*(float x)
{
	return MAD(x, 0.f);
}

TensorCL TensorCL::operator/(float x)
{
	return MAD(1.f/x, 0.f);
}

TensorCL TensorCL::operator>(float x)
{
	TensorCL C(param); //create a temporary array
	more_n.SetRange(CL->group_size[0], 1, param.length, 1);
	more_n.SetArg(0, C.data); //result
	more_n.SetArg(1, data);
	more_n.SetArg(2, param);
	more_n.SetArg(3, x);
	more_n.RFlush();
	return C;
}

TensorCL TensorCL::operator<(float x)
{
	TensorCL C(param); //create a temporary array
	less_n.SetRange(CL->group_size[0], 1, param.length, 1);
	less_n.SetArg(0, C.data); //result
	less_n.SetArg(1, data);
	less_n.SetArg(2, param);
	less_n.SetArg(3, x);
	less_n.RFlush();
	return C;
}

TensorCL TensorCL::sin()
{
	TensorCL C(param); //create a temporary array
	sinfun.SetRange(CL->group_size[0], 1, param.length, 1);
	sinfun.SetArg(0, C.data); //result
	sinfun.SetArg(1, data);
	sinfun.SetArg(2, param);
	sinfun.RFlush();
	return C;
}

TensorCL TensorCL::cos()
{
	TensorCL C(param); //create a temporary array
	cosfun.SetRange(CL->group_size[0], 1, param.length, 1);
	cosfun.SetArg(0, C.data); //result
	cosfun.SetArg(1, data);
	cosfun.SetArg(2, param);
	cosfun.RFlush();
	return C;
}

TensorCL TensorCL::tan()
{
	TensorCL C(param); //create a temporary array
	tanfun.SetRange(CL->group_size[0], 1, param.length, 1);
	tanfun.SetArg(0, C.data); //result
	tanfun.SetArg(1, data);
	tanfun.SetArg(2, param);
	tanfun.RFlush();
	return C;
}

TensorCL TensorCL::exp()
{
	TensorCL C(param); //create a temporary array
	expfun.SetRange(CL->group_size[0], 1, param.length, 1);
	expfun.SetArg(0, C.data); //result
	expfun.SetArg(1, data);
	expfun.SetArg(2, param);
	expfun.RFlush();
	return C;
}

TensorCL TensorCL::log()
{
	TensorCL C(param); //create a temporary array
	logfun.SetRange(CL->group_size[0], 1, param.length, 1);
	logfun.SetArg(0, C.data); //result
	logfun.SetArg(1, data);
	logfun.SetArg(2, param);
	logfun.RFlush();
	return C;
}

TensorCL TensorCL::tanh()
{
	TensorCL C(param); //create a temporary array
	tanhfun.SetRange(CL->group_size[0], 1, param.length, 1);
	tanhfun.SetArg(0, C.data); //result
	tanhfun.SetArg(1, data);
	tanhfun.SetArg(2, param);
	tanhfun.RFlush();
	return C;
}

TensorCL TensorCL::operator^(float y)
{
	TensorCL C(param); //create a temporary array
	powfun.SetRange(CL->group_size[0], 1, param.length, 1);
	powfun.SetArg(0, C.data); //result
	powfun.SetArg(1, data);
	powfun.SetArg(2, param);
	powfun.SetArg(3, y);
	powfun.RFlush();
	return C;
}

TensorCL TensorCL::dot(TensorCL & X)
{
	if (AreTensorsCompatible(param, X.param))
	{
		TensorCL C(TensorDotResult(param, X.param)); //create a temporary array

		m_dot.SetRange(TS, TS, C.param.size[0], C.param.size[1]);
		m_dot.SetArg(0, C.data); //result
		m_dot.SetArg(1, data);
		m_dot.SetArg(2, X.data);
		m_dot.SetArg(3, C.param);
		m_dot.SetArg(4, param);
		m_dot.SetArg(5, X.param);

		int shiftA = 0;
		int shiftB = 0;
		for (int shiftC = 0; shiftC < C.param.length; shiftC += C.param.size[0] * C.param.size[1])
		{
			m_dot.SetArg(6, shiftA);
			m_dot.SetArg(7, shiftB);
			m_dot.SetArg(8, shiftC);
			m_dot.RFlush();

			shiftA = (shiftA + this->param.size[0] * this->param.size[1]) % this->GetLength();
			shiftB = (shiftB + X.param.size[0] * X.param.size[1]) % X.GetLength();
		}

		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

void TensorCL::LoadData(float * data_ptr)
{
}

float * TensorCL::GetData()
{
	clEnqueueReadBuffer(CL->queue(), data, CL_TRUE, 0, sizeof(float)*param.length, host_data, NULL, NULL, NULL);
	return host_data; 
}

int TensorCL::GetLength()
{
	return param.length;
}

cl_tensor TensorCL::GetParam()
{
	return param;
}

cl_tensor Transpose(cl_tensor x, int dim_a, int dim_b)
{
	int buf = x.size[dim_a];
	x.size[dim_a] = x.size[dim_b];
	x.size[dim_b] = buf;
	return x;
}

cl_tensor GetSumTensor(cl_tensor x)
{
	x.length = x.length / x.size[x.rank - 1];
	x.size[x.rank-1] = 1;
	x.rank -= 1;
	return x;
}

cl_tensor Repeat(cl_tensor x, int n)
{
	x.length = x.length * n;
	x.size[x.rank] = n;
	x.rank += 1;
	return x;
}

void TensorUseOpenCL(OpenCL *cl)
{
	CL = cl;
	add.Initialize("tensor_add", CL);
	mul.Initialize("tensor_mul", CL);
	mad.Initialize("tensor_mad", CL);
	m_dot.Initialize("tensor_dot_product", CL);
	idx.Initialize("tensor_index", CL);
	sinfun.Initialize("tensor_sin", CL);
	cosfun.Initialize("tensor_cos", CL);
	tanfun.Initialize("tensor_tan", CL);
	minfun.Initialize("tensor_min", CL);
	maxfun.Initialize("tensor_max", CL);
	maxfun_f.Initialize("tensor_max_f", CL);
	minfun_f.Initialize("tensor_min_f", CL);
	expfun.Initialize("tensor_exp", CL);
	logfun.Initialize("tensor_log", CL);
	transposefun.Initialize("tensor_transpose", CL); 
	sumfun.Initialize("tensor_sum", CL);
	tanhfun.Initialize("tensor_tanh", CL);
	repeatfun.Initialize("tensor_repeat", CL);
	more_m.Initialize("tensor_more_m", CL);
	less_m.Initialize("tensor_less_m", CL);
	more_n.Initialize("tensor_more_n", CL);
	less_n.Initialize("tensor_less_n", CL);
	if_cond.Initialize("tensor_if", CL);
}

void PrintTensor(TensorCL & a)
{
	float* hd = a.GetData();
	cl_tensor T = a.GetParam();
	std::cout << "=====================================" << std::endl;
	for (int i = 0; i < T.rank; i++)
	{
		std::cout << T.size[i];
		if (i < T.rank - 1)
		{
			std::cout << "*";
		}
	}
	std::cout << std::endl;
	for (int shift = 0; shift < T.length; shift += T.size[0] * T.size[1])
	{
		for (int i = 0; i < T.size[0]; i++)
		{
			for (int j = 0; j < T.size[1]; j++)
			{
				std::cout << hd[shift + T.size[0]*j + i] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
}

TensorCL TensorCL::sum()
{
	TensorCL C(GetSumTensor(GetParam())); //create a temporary array
	sumfun.SetRange(CL->group_size[0], 1, C.GetParam().length, 1);
	sumfun.SetArg(0, C.data); //result
	sumfun.SetArg(1, data);
	sumfun.SetArg(2, C.GetParam());
	sumfun.SetArg(3, GetParam());
	sumfun.RFlush();
	return C;
}

TensorCL TensorCL::min(TensorCL &X)
{
	TensorCL C(GetParam()); //create a temporary array
	minfun.SetRange(CL->group_size[0], 1, param.length, 1);
	minfun.SetArg(0, C.data); //result
	minfun.SetArg(1, data);
	minfun.SetArg(2, X.data);
	minfun.SetArg(3, GetParam());
	minfun.RFlush();
	return C;
}

TensorCL TensorCL::max(TensorCL &X)
{
	TensorCL C(GetParam()); //create a temporary array
	maxfun.SetRange(CL->group_size[0], 1, param.length, 1);
	maxfun.SetArg(0, C.data); //result
	maxfun.SetArg(1, data);
	maxfun.SetArg(2, X.data);
	maxfun.SetArg(3, GetParam());
	maxfun.RFlush();
	return C;
}

TensorCL TensorCL::min(float y)
{
	TensorCL C(GetParam()); //create a temporary array
	minfun_f.SetRange(CL->group_size[0], 1, param.length, 1);
	minfun_f.SetArg(0, C.data); //result
	minfun_f.SetArg(1, data);
	minfun_f.SetArg(2, GetParam());
	minfun_f.SetArg(3, y);
	minfun_f.RFlush();
	return C;
}

TensorCL TensorCL::max(float y)
{
	TensorCL C(GetParam()); //create a temporary array
	maxfun_f.SetRange(CL->group_size[0], 1, param.length, 1);
	maxfun_f.SetArg(0, C.data); //result
	maxfun_f.SetArg(1, data);
	maxfun_f.SetArg(2, GetParam());
	maxfun_f.SetArg(3, y);
	maxfun_f.RFlush();
	return C;
}

TensorCL TensorCL::_if(TensorCL & _true, TensorCL & _false)
{
	if (AreTensorsEqual(param, _true.param) && AreTensorsEqual(param, _false.param))
	{
		TensorCL C(param); //create a temporary array
		if_cond.SetRange(CL->group_size[0], 1, param.length, 1);
		if_cond.SetArg(0, C.data); //result
		if_cond.SetArg(1, data);
		if_cond.SetArg(2, _true.data);
		if_cond.SetArg(3, _false.data);
		if_cond.SetArg(4, param);
		if_cond.RFlush();
		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

TensorCL TensorCL::indicies(int dim)
{
	TensorCL C(GetParam());
	idx.SetRange(CL->group_size[0], 1, param.length, 1);
	idx.SetArg(0, C.data); //result
	idx.SetArg(1, GetParam());
	idx.SetArg(2, dim);
	idx.RFlush();
	return C;
}

void TensorCL::reshape(int x, int y, int z, int w)
{
	
}

TensorCL TensorCL::transpose(int dim_a, int dim_b)
{
	TensorCL C(Transpose(GetParam(), dim_a, dim_b));
	transposefun.SetRange(CL->group_size[0], 1, param.length, 1);
	transposefun.SetArg(0, C.data); //result
	transposefun.SetArg(1, data);
	transposefun.SetArg(2, C.GetParam());
	transposefun.SetArg(3, GetParam());
	transposefun.SetArg(4, dim_a);
	transposefun.SetArg(5, dim_b);
	transposefun.RFlush();
	return C;
}

TensorCL TensorCL::repeat(int n)
{
	TensorCL C(Repeat(GetParam(), n));
	repeatfun.SetRange(CL->group_size[0], 1, param.length, 1);
	repeatfun.SetArg(0, C.data); //result
	repeatfun.SetArg(1, data);
	repeatfun.SetArg(2, C.GetParam());
	repeatfun.SetArg(3, GetParam());
	repeatfun.SetArg(4, n);
	repeatfun.RFlush();
	return C;
}

TensorCL TensorCL::MAD(float a, float b)
{
	TensorCL C(GetParam()); //create a temporary array
	mad.SetRange(CL->group_size[0], 1, param.length, 1);
	mad.SetArg(0, C.data); //result
	mad.SetArg(1, data);
	mad.SetArg(2, GetParam());
	mad.SetArg(3, a);
	mad.SetArg(4, b);
	mad.RFlush();
	return C;
}

bool AreTensorsEqual(cl_tensor x, cl_tensor y)
{
	bool equal = true;
	for (int i = 0; i < MAX_DIM; i++)
	{
		equal = equal && (x.size[i] == y.size[i]);
	}
	equal = equal && (x.rank == y.rank);
	equal = equal && (x.length == y.length);
	return equal;
}

bool AreTensorsCompatible(cl_tensor x, cl_tensor y)
{
	bool comp = true;
	comp = comp && (x.rank <= y.rank);

	for (int i = 2; i < x.rank; i++)
	{
		comp = comp && (x.size[i] == y.size[i] || x.size[i] == 0 || x.size[i] == 1);
	}
	
	comp = comp && (x.size[1] == y.size[0]);
	return comp;
}

cl_tensor TensorDotResult(cl_tensor x, cl_tensor y)
{
	cl_tensor res = y;
	res.size[0] = x.size[0];

	res.length = 1;
	for (int i = 0; i < res.rank; i++)
	{
		res.length *= res.size[i];
	}
	return res;
}

int GetIndex(cl_index id, cl_tensor param)
{
	return 0;
}
