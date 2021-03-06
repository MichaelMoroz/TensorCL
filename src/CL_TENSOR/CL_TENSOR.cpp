#include "CL_TENSOR.h"

OpenCL *CL;
CLFunction add, mul, mad, m_dot;
CLFunction more_m, less_m;
CLFunction more_n, less_n, if_cond;
CLFunction idx, sinfun, cosfun, tanfun, tanhfun, powfun, maxfun, minfun, maxfun_f, minfun_f;
CLFunction expfun, logfun, transposefun, repeatfun;
CLFunction sumfun, randfun, diagfun, cutfun;

std::string GetTensorSizeString(cl_tensor x)
{
	std::string info = "(";
	for (int i = 0; i < x.rank; i++)
	{
		info = info + num_to_str(x.size[i]) + ((i!=x.rank-1)?",":"");
	}
	info = info + ")";
	return info;
}

TensorCL::TensorCL()
{
	data = NULL;
}

void TensorCL::release()
{
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
	for (int i = 0; i < MAX_DIM; i++)
	{
		param.length *= param.size[i];
	}
	cl_int status;
	param.rank = getrank(param);
	data = clCreateBuffer(CL->default_context(), CL_MEM_READ_WRITE, param.length * sizeof(float), 0, &status);

	if (status != CL_SUCCESS)
	{
		std::string err = "OpenCL error: " + getOpenCLError(status);
		ERROR_MSG(err.c_str());
	}
	else
	{
		clEnqueueFillBuffer(CL->queue(), data, &value, sizeof(value), 0, param.length * sizeof(float), 0, 0, 0);
	}
}

TensorCL::TensorCL(Size s, float fill, bool rand)
{
	param = s.param;
	init_data(fill);
	if (rand)
	{
		*this = this->random()*fill;
	}
}

TensorCL::TensorCL(int r, std::vector<int> s)
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

TensorCL::TensorCL(std::ifstream &file)
{
	LoadFromFstream(file);
}

TensorCL::TensorCL(cl_tensor p) : param(p)
{
	init_data();
}


TensorCL& TensorCL::operator=(float a)
{
 	*this = MAD(0, a);
	return *this;
}

//r-value -> move 
TensorCL::TensorCL(TensorCL && p)
{
	std::swap(this->data, p.data);
	std::swap(this->param, p.param);
	p.data = NULL;
}

TensorCL::TensorCL(TensorData & D)
{
	param = D.GetParam();
	init_data();
	clEnqueueWriteBuffer(CL->queue(), data, CL_TRUE, 0, sizeof(float)*param.length, D.GetData(), NULL, NULL, NULL);
}

TensorCL & TensorCL::operator=(TensorCL && p)
{
	release();
	std::swap(this->data, p.data);
	std::swap(this->param, p.param);
	p.data = NULL;
	return *this;
}

//l-value -> copy
TensorCL& TensorCL::operator=(const TensorCL & X)
{
	release();
	param = X.param;
	init_data();
	clEnqueueCopyBuffer(CL->queue(), X.data, data, 0, 0, param.length * sizeof(float), NULL, NULL, NULL);
	CL->queue.flush();
	return *this;
}

TensorCL::TensorCL(const TensorCL &X)
{
	param = X.param;
	//if array is temporary - then just copy the CL pointer
	init_data();
	clEnqueueCopyBuffer(CL->queue(), data, X.data, 0, 0, param.length * sizeof(float), NULL, NULL, NULL);
	CL->queue.flush();
}

TensorCL& TensorCL::operator+=(TensorCL &X)
{
	return *this = *this + X;
}

TensorCL& TensorCL::operator-=(TensorCL &X)
{
	return *this = *this - X;
}

TensorCL& TensorCL::operator*=(TensorCL &X)
{
	return *this = *this * X;
}

TensorCL& TensorCL::operator/=(TensorCL &X)
{
	return *this = *this / X;
}

TensorCL & TensorCL::operator+=(float & X)
{
	return *this = *this + X;
}

TensorCL & TensorCL::operator-=(float & X)
{
	return *this = *this - X;
}

TensorCL & TensorCL::operator*=(float & X)
{
	return *this = *this * X;
}

TensorCL & TensorCL::operator/=(float & X)
{
	return *this = *this / X;
}

TensorCL::TensorCL(TensorCL &X, float fill)
{
	param = X.param;
	init_data(fill);
}

TensorCL TensorCL::operator+(TensorCL & X)
{
	if (AreTensorsEqual(GetParam(), X.GetParam()))
	{
		TensorCL C(GetParam()); //create a temporary array
		add.SetRange(CL->group_size[0], 1, param.length, 1);
		add.SetArg(0, C.data); //result
		add.SetArg(1, data);
		add.SetArg(2, X.data);
		add.SetArg(3, GetParam());
		add.SetArg(4, 1.f);
		add.SetArg(5, 1.f);
		add.RFlush();
		return C;
	}
	else
	{
		ERROR_MSG(("ADD: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = " + GetTensorSizeString(X.GetParam())).c_str());
	}
	return *this;
}

TensorCL TensorCL::operator-(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(GetParam()); //create a temporary array
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
		ERROR_MSG(("SUBS: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = " + GetTensorSizeString(X.GetParam())).c_str());
	}
	return *this;
}

TensorCL TensorCL::operator*(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(GetParam()); //create a temporary array
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
		ERROR_MSG(("MUL: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = " + GetTensorSizeString(X.GetParam())).c_str());
	}
	return *this;
}

TensorCL TensorCL::operator/(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(GetParam()); //create a temporary array
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
		ERROR_MSG(("DIV: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = " + GetTensorSizeString(X.GetParam())).c_str());
	}
	return *this;
}

TensorCL TensorCL::operator>(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(GetParam()); //create a temporary array
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
		ERROR_MSG(("MORE: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = " + GetTensorSizeString(X.GetParam())).c_str());
	}
	return *this;
}

TensorCL TensorCL::operator<(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(GetParam()); //create a temporary array
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
		ERROR_MSG(("LESS: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = " + GetTensorSizeString(X.GetParam())).c_str());
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
	TensorCL C(GetParam()); //create a temporary array
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
	TensorCL C(GetParam()); //create a temporary array
	less_n.SetRange(CL->group_size[0], 1, param.length, 1);
	less_n.SetArg(0, C.data); //result
	less_n.SetArg(1, data);
	less_n.SetArg(2, param);
	less_n.SetArg(3, x);
	less_n.RFlush();
	return C;
}


TensorCL TensorCL::random()
{
	TensorCL C(GetParam()); //create a temporary array
	randfun.SetRange(CL->group_size[0], 1, param.length, 1);
	randfun.SetArg(0, C.data); //result
	randfun.SetArg(1, param);
	randfun.SetArg(2, rand());
	randfun.RFlush();
	return C;
}

TensorCL TensorCL::sin()
{
	TensorCL C(GetParam()); //create a temporary array
	sinfun.SetRange(CL->group_size[0], 1, param.length, 1);
	sinfun.SetArg(0, C.data); //result
	sinfun.SetArg(1, data);
	sinfun.SetArg(2, param);
	sinfun.RFlush();
	return C;
}

TensorCL TensorCL::cos()
{
	TensorCL C(GetParam()); //create a temporary array
	cosfun.SetRange(CL->group_size[0], 1, param.length, 1);
	cosfun.SetArg(0, C.data); //result
	cosfun.SetArg(1, data);
	cosfun.SetArg(2, param);
	cosfun.RFlush();
	return C;
}

TensorCL TensorCL::tan()
{
	TensorCL C(GetParam()); //create a temporary array
	tanfun.SetRange(CL->group_size[0], 1, param.length, 1);
	tanfun.SetArg(0, C.data); //result
	tanfun.SetArg(1, data);
	tanfun.SetArg(2, param);
	tanfun.RFlush();
	return C;
}

TensorCL TensorCL::exp()
{
	TensorCL C(GetParam()); //create a temporary array
	expfun.SetRange(CL->group_size[0], 1, param.length, 1);
	expfun.SetArg(0, C.data); //result
	expfun.SetArg(1, data);
	expfun.SetArg(2, param);
	expfun.RFlush();
	return C;
}

TensorCL TensorCL::log()
{
	TensorCL C(GetParam()); //create a temporary array
	logfun.SetRange(CL->group_size[0], 1, param.length, 1);
	logfun.SetArg(0, C.data); //result
	logfun.SetArg(1, data);
	logfun.SetArg(2, param);
	logfun.RFlush();
	return C;
}

TensorCL TensorCL::tanh()
{
	TensorCL C(GetParam()); //create a temporary array
	tanhfun.SetRange(CL->group_size[0], 1, param.length, 1);
	tanhfun.SetArg(0, C.data); //result
	tanhfun.SetArg(1, data);
	tanhfun.SetArg(2, param);
	tanhfun.RFlush();
	return C;
}

TensorCL TensorCL::pow(float y)
{
	TensorCL C(GetParam()); //create a temporary array
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
	if (AreTensorsCompatible(GetParam(), X.GetParam()))
	{
		TensorCL C(TensorDotResult(GetParam(), X.GetParam())); //create a temporary array

		m_dot.SetRangeExactly(TS, TS, C.param.size[0], C.param.size[1]);
		m_dot.SetArg(0, C.data); //result
		m_dot.SetArg(1, data);
		m_dot.SetArg(2, X.data);
		m_dot.SetArg(3, C.GetParam());
		m_dot.SetArg(4, GetParam());
		m_dot.SetArg(5, X.GetParam());

		int shiftA = 0;
		int shiftB = 0;
		for (int shiftC = 0; shiftC < C.GetLength(); shiftC += C.param.size[0] * C.param.size[1])
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
		ERROR_MSG(("DOT: Incompatible tensors, A = " + GetTensorSizeString(GetParam()) + ", B = "+ GetTensorSizeString(X.GetParam())).c_str());
	}
	return *this;
}

void TensorCL::LoadData(float * data_ptr)
{
}

float * TensorCL::GetData()
{
	float* host_data = new float[param.length];
	clEnqueueReadBuffer(CL->queue(), data, CL_TRUE, 0, sizeof(float)*param.length, host_data, NULL, NULL, NULL);
	return host_data; 
}


int ID(cl_index indx, cl_tensor info)
{
	int id = 0;
	int sh = 1;
	for (int i = 0; i < info.rank; i++)
	{
		id += indx.index[i] * sh;
		sh *= info.size[i];
	}
	return id;
}

float TensorCL::operator()(int i, int j, int k, int m)
{
	float value;
	cl_index id;
	id.index[0] = i;
	id.index[1] = j;
	id.index[2] = k;
	id.index[3] = m;
	//copy a single value from the GPU
	clEnqueueReadBuffer(CL->queue(), data, CL_TRUE, ID(id, param), sizeof(float), &value, NULL, NULL, NULL);
	return value;
}

void TensorCL::LoadFromFstream(std::ifstream &file)
{
	cl_tensor header;
	file.read((char*)&header, sizeof(cl_tensor));
	TensorData D(header);
	file.read((char*)D.GetData(), header.length*sizeof(float));
	param = header;
	init_data(0.f);
	clEnqueueWriteBuffer(CL->queue(), data, CL_TRUE, 0, sizeof(float)*param.length, D.GetData(), NULL, NULL, NULL);
}

void TensorCL::SaveToFstream(std::ofstream &file)
{
	file.write((char*)&param, sizeof(cl_tensor));
	float* data = GetData();
	file.write((char*)data, param.length * sizeof(float));
	delete[] data;
}

int TensorCL::GetLength()
{
	return param.length;
}

cl_tensor TensorCL::GetParam()
{
	param.rank = getrank(param);
	return param;
}

Size TensorCL::GetSize()
{
	Size size;
	size.param = this->param;
	return size;
}

int getrank(cl_tensor x)
{
	int rank = 1;
	for (int i = 0; i < MAX_DIM; i++)
	{
		if (x.size[i] > 1)
		{
			rank = i + 1;
		}
	}
	return rank;
}

cl_tensor Transpose(cl_tensor x, int dim_a, int dim_b)
{
	x.rank = getrank(x);
	int buf = x.size[dim_a];
	x.size[dim_a] = x.size[dim_b];
	x.size[dim_b] = buf;
	if (std::max(dim_a, dim_b) > x.rank)
	{
		x.rank = std::max(dim_a, dim_b);
	}
	return x;
}

cl_tensor GetSumTensor(cl_tensor x)
{
	x.rank = getrank(x);
	if (x.rank == 1)
	{
		x.length = 1;
		x.size[x.rank - 1] = 1;
	}
	else
	{
		x.length = x.length / x.size[x.rank - 1];
		x.size[x.rank - 1] = 1;
		x.rank -= 1;
	}
	return x;
}

cl_tensor UpdateLength(cl_tensor x)
{
	x.length = 1;
	for (int i = 0; i < MAX_DIM; i++)
	{
		x.length *= x.size[i];
	}
	return x;
}

cl_tensor Repeat(cl_tensor x, int n)
{
	x.rank = getrank(x);
	x.length = x.length * n;
	x.size[x.rank] = n;
	x.rank += 1;
	return x;
}

cl_tensor Cut(cl_tensor x, int a, int b)
{
	x.rank = getrank(x);
	x.length = x.length / x.size[x.rank - 1];
	x.size[x.rank - 1] = b - a + 1;//if cut from a to a then size = 1
	x.length = x.length * x.size[x.rank - 1];
	x.rank = getrank(x);
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
	powfun.Initialize("tensor_pow", CL); 
	randfun.Initialize("tensor_random", CL);
	diagfun.Initialize("tensor_diag", CL);
	cutfun.Initialize("tensor_cut", CL);
}

inline std::ostream& red(std::ostream &s)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hStdout,
		FOREGROUND_RED | FOREGROUND_INTENSITY);
	return s;
}

inline std::ostream& white(std::ostream &s)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hStdout,
		FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
	return s;
}

inline std::ostream& yellow(std::ostream &s)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hStdout,
		FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
	return s;
}

TensorCL reshape(TensorCL & X, int x, int y, int z, int w)
{
	TensorCL C = X;
	C.reshape(x, y, z, w);
	return C;
}

TensorCL reshape(TensorCL & X, cl_tensor new_param)
{
	TensorCL C = X;
	C.reshape(new_param);
	return C;
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
				float val = hd[shift + T.size[0] * j + i];
				if (!isfinite(val))
				{
					std::cout << red;
				}
				else if(abs(val)>1e5)
				{
					std::cout << yellow;
				}
				std::cout << hd[shift + T.size[0]*j + i] << " ";
				std::cout << white;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	delete[] hd;
}

TensorCL TensorCL::sum()
{
	TensorCL C(GetSumTensor(GetParam())); //create a temporary array
	sumfun.SetRange(CL->group_size[0]/8, 1, C.GetLength(), 1);
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

TensorCL TensorCL::diag(float x, float y)
{
	TensorCL C(GetParam()); //create a temporary array
	diagfun.SetRange(CL->group_size[0], 1, param.length, 1);
	diagfun.SetArg(0, C.data); //result
	diagfun.SetArg(1, GetParam());
	diagfun.SetArg(2, x);
	diagfun.SetArg(3, y);
	diagfun.RFlush();
	return C;
}

TensorCL TensorCL::_if(TensorCL & _true, TensorCL & _false)
{
	if (AreTensorsEqual(GetParam(), _true.GetParam()) && AreTensorsEqual(GetParam(), _false.GetParam()))
	{
		TensorCL C(GetParam()); //create a temporary array
		if_cond.SetRange(CL->group_size[0], 1, param.length, 1);
		if_cond.SetArg(0, C.data); //result
		if_cond.SetArg(1, data);
		if_cond.SetArg(2, _true.data);
		if_cond.SetArg(3, _false.data);
		if_cond.SetArg(4, GetParam());
		if_cond.RFlush();
		return C;
	}
	else
	{
		ERROR_MSG("IF: Incompatible tensors");
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
	cl_tensor newp = GetParam();
	newp.size[0] = x;
	newp.size[1] = y;
	newp.size[2] = z;
	newp.size[3] = w;
	newp = UpdateLength(newp);
	if (newp.length == param.length)
	{
		param = newp;
	}
	else
	{
		ERROR_MSG("RESHAPE: Incorrect parameters");
	}
}

void TensorCL::reshape(cl_tensor new_param)
{
	new_param = UpdateLength(new_param);
	if (new_param.length == param.length)
	{
		param = new_param;
	}
	else
	{
		ERROR_MSG("RESHAPE: Incorrect parameters");
	}
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
	repeatfun.SetRange(CL->group_size[0], 1, C.GetLength(), 1);
	repeatfun.SetArg(0, C.data); //result
	repeatfun.SetArg(1, data);
	repeatfun.SetArg(2, C.GetParam());
	repeatfun.SetArg(3, GetParam());
	repeatfun.SetArg(4, n);
	repeatfun.RFlush();
	return C;
}

TensorCL TensorCL::cut(int from, int to)
{
	TensorCL C(Cut(GetParam(),from,to)); //create a temporary array
	cutfun.SetRange(CL->group_size[0], 1, C.GetLength(), 1);
	cutfun.SetArg(0, C.data); //result
	cutfun.SetArg(1, data);
	cutfun.SetArg(2, C.GetParam());
	cutfun.SetArg(3, GetParam());
	cutfun.SetArg(4, from);
	cutfun.SetArg(5, to);
	cutfun.RFlush();
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
	equal = equal && (x.length == y.length);
	return equal;
}

bool AreTensorsCompatible(cl_tensor x, cl_tensor y)
{
	bool comp = true;
	comp = comp && (getrank(x) <= getrank(y) || (getrank(y) == 1 && getrank(x) == 2));

	for (int i = 2; i < getrank(x); i++)
	{
		comp = comp && (x.size[i] == y.size[i] || x.size[i] == 0 || x.size[i] == 1);
	}
	
	comp = comp && (x.size[1] == y.size[0]);
	return comp;
}

cl_tensor TensorDotResult(cl_tensor x, cl_tensor y)
{
	x.rank = getrank(x);
	y.rank = getrank(y);

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

TensorData::TensorData(int x, int y, int z, int w)
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

	param.length = 1;
	for (int i = 0; i < param.rank; i++)
	{
		param.length *= param.size[i];
	}

	data.reset(new float[param.length]);
	std::fill(data.get(), data.get() + param.length, 0.f);
}

TensorData::TensorData(cl_tensor P) 
{
	param = P;
	data.reset(new float[param.length]);
	std::fill(data.get(), data.get() + param.length, 0.f); 
}

void TensorData::LoadData(std::vector< std::vector< std::vector<float> > > A)
{
	if (A[0][0].size() == param.size[0] && A[0].size() == param.size[1] && A.size() == param.size[2])
	{
		int pos = 0;
		for (auto i = 0; i < A.size(); i++)
		{
			for (auto j = 0; j < A[i].size(); j++)
			{
				std::copy(A[i][j].begin(), A[i][j].end(), data.get()+pos);
				pos += A[i][j].size();
			}
		}
	}
	else
	{
		ERROR_MSG("Invalid data size");
	}
}

void TensorData::LoadData(std::vector< std::vector<float> > A)
{
	if (A.size() == param.size[1] && A[0].size() == param.size[0])
	{
		for (auto i = 0; i < A.size(); i++)
		{
			std::copy(A[i].begin(), A[i].end(), data.get() + A[i].size()*i);
		}
	}
	else
	{
		ERROR_MSG("Invalid data size");
	}
}

void TensorData::LoadData(std::vector<float> A)
{
	if (A.size() == param.size[0])
	{
		std::copy(A.begin(), A.end(), data.get());
	}
	else
	{
		ERROR_MSG("Invalid data size");
	}
}

cl_tensor TensorData::GetParam()
{
	param.rank = getrank(param);
	return param;
}

float * TensorData::GetData()
{
	return data.get();
}

Size::Size(int x, int y, int z, int w)
{
	param.size[0] = x;
	param.size[1] = y;
	param.size[2] = z;
	param.size[3] = w;

	param.length = 1;
	for (int i = 0; i < param.rank; i++)
	{
		param.length *= param.size[i];
	}

	param.rank = getrank(param);
}
