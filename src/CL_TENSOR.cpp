#include "CL_TENSOR.h"

OpenCL *CL;
CLFunction add, mul, mad, m_dot;
CLFunction idx, sinfun, cosfun, tanfun, powfun, maxfun, minfun, maxfun_f, minfun_f;
CLFunction expfun, logfun;

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

void TensorCL::init_data()
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
		float value = 0;
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

TensorCL::TensorCL(int x)
{
	param.size[0] = x;
	param.rank = 2;
	init_data();
}

TensorCL::TensorCL(int x, int y)
{
	param.size[0] = x;
	param.size[1] = y;
	param.rank = 2;
	init_data();
}

TensorCL::TensorCL(int x, int y, int z)
{
	param.size[0] = x;
	param.size[1] = y;
	param.size[2] = z;
	param.rank = 3;
	init_data();
}

TensorCL::TensorCL(int x, int y, int z, int w) 
{
	param.size[0] = x;
	param.size[1] = y;
	param.size[2] = z;
	param.size[3] = w;
	param.rank = 4;
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
	clEnqueueCopyBuffer(CL->queue(), data, X.data, 0, 0, param.length * sizeof(float), NULL, NULL, NULL);
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

		for (int shift = 0; shift < C.param.length; shift += C.param.size[C.param.rank - 2] * C.param.size[C.param.rank - 1])
		{
			m_dot.SetRange(TS, TS, C.param.size[C.param.rank - 2], C.param.size[C.param.rank - 1]);
			m_dot.SetArg(0, C.data); //result
			m_dot.SetArg(1, data);
			m_dot.SetArg(2, X.data);
			m_dot.SetArg(3, C.param);
			m_dot.SetArg(4, param);
			m_dot.SetArg(5, X.param);
			m_dot.SetArg(6, shift);
			m_dot.RFlush();
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
}

void PrintTensor(TensorCL & a)
{
	float* hd = a.GetData();
	cl_tensor T = a.GetParam();
	for (int shift = 0; shift < T.length; shift += T.size[T.rank - 2] * T.size[T.rank - 1])
	{
		for (int i = 0; i < T.size[T.rank - 2]; i++)
		{
			for (int j = 0; j < T.size[T.rank - 1]; j++)
			{
				std::cout << hd[shift + T.size[T.rank - 2]*i + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
}

TensorCL operator+(float x, TensorCL & Y)
{
	return Y + x;
}

TensorCL operator-(float x, TensorCL & Y)
{
	return -Y + x;
}

TensorCL TensorCL::min(TensorCL &X)
{
	TensorCL C(param); //create a temporary array
	minfun.SetRange(CL->group_size[0], 1, param.length, 1);
	minfun.SetArg(0, C.data); //result
	minfun.SetArg(1, data);
	minfun.SetArg(2, X.data);
	minfun.SetArg(3, param);
	minfun.RFlush();
	return C;
}

TensorCL TensorCL::max(TensorCL &X)
{
	TensorCL C(param); //create a temporary array
	maxfun.SetRange(CL->group_size[0], 1, param.length, 1);
	maxfun.SetArg(0, C.data); //result
	maxfun.SetArg(1, data);
	maxfun.SetArg(2, X.data);
	maxfun.SetArg(3, param);
	maxfun.RFlush();
	return C;
}

TensorCL TensorCL::min(float y)
{
	TensorCL C(param); //create a temporary array
	minfun_f.SetRange(CL->group_size[0], 1, param.length, 1);
	minfun_f.SetArg(0, C.data); //result
	minfun_f.SetArg(1, data);
	minfun_f.SetArg(2, param);
	minfun_f.SetArg(3, y);
	minfun_f.RFlush();
	return C;
}

TensorCL TensorCL::max(float y)
{
	TensorCL C(param); //create a temporary array
	maxfun_f.SetRange(CL->group_size[0], 1, param.length, 1);
	maxfun_f.SetArg(0, C.data); //result
	maxfun_f.SetArg(1, data);
	maxfun_f.SetArg(2, param);
	maxfun_f.SetArg(3, y);
	maxfun_f.RFlush();
	return C;
}

TensorCL TensorCL::indicies()
{
	TensorCL C(param);
	idx.SetRange(CL->group_size[0], 1, param.length, 1);
	idx.SetArg(0, C.data); //result
	idx.SetArg(1, param);
	idx.RFlush();
	return C;
}

TensorCL TensorCL::MAD(float a, float b)
{
	TensorCL C(param); //create a temporary array
	mad.SetRange(CL->group_size[0], 1, param.length, 1);
	mad.SetArg(0, C.data); //result
	mad.SetArg(1, data);
	mad.SetArg(2, param);
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
	comp = comp && (x.rank == y.rank);

	for (int i = 0; i < x.rank - 2; i++)
	{
		comp = comp && (x.size[i] == y.size[i]);
	}
	
	comp = comp && (x.size[x.rank - 1] == y.size[y.rank - 1 - 1]);
	return comp;
}

cl_tensor TensorDotResult(cl_tensor x, cl_tensor y)
{
	cl_tensor res = x;
	res.size[res.rank - 1] = y.size[res.rank - 1];
	//res.size[res.rank - 1 - 1] = x.size[res.rank - 1 - 1];
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

TensorCL operator*(float x, TensorCL & Y)
{
	return Y*x;
}

TensorCL operator/(float x, TensorCL & Y)
{
	return Y/x;
}

TensorCL sin(TensorCL & X)
{
	return X.sin();
}

TensorCL cos(TensorCL & X)
{
	return X.cos();
}

TensorCL tan(TensorCL & X)
{
	return X.tan();
}

TensorCL exp(TensorCL & X)
{
	return X.exp();
}

TensorCL log(TensorCL & X)
{
	return X.log();
}

TensorCL min(TensorCL & X, TensorCL & Y)
{
	return X.min(Y);
}

TensorCL max(TensorCL & X, TensorCL & Y)
{
	return X.max(Y);
}

TensorCL min(TensorCL & X, float y)
{
	return X.min(y);
}

TensorCL max(TensorCL & X, float y)
{
	return X.max(y);
}

TensorCL min(float y, TensorCL & X)
{
	return X.max(y);
}

TensorCL max(float y, TensorCL & X)
{
	return X.min(y);
}

TensorCL dot(TensorCL & X, TensorCL & Y)
{
	return X.dot(Y);
}

TensorCL indicies(TensorCL & X)
{
	return X.indicies();
}
