#include "CL_TENSOR.h"

OpenCL *CL;
CLFunction add, mul, mad, m_dot;

bool TensorCL::isTemporary()
{
	return temporary;
}

void TensorCL::release()
{
	clReleaseMemObject(data);
}

TensorCL::~TensorCL()
{
	if (!temporary)
	{
		release();
	}
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
}

TensorCL::TensorCL(int r, vector<int> s, bool temp) : temporary(temp)
{
	param.rank = r;
	std::copy(s.begin(), s.end(), param.size);
	init_data();
}

TensorCL::TensorCL(int x, bool temp) : temporary(temp)
{
	param.size[0] = x;
	param.rank = 2;
	init_data();
}

TensorCL::TensorCL(int x, int y, bool temp) :  temporary(temp)
{
	param.size[0] = x;
	param.size[1] = y;
	param.rank = 2;
	init_data();
}

TensorCL::TensorCL(int x, int y, int z, bool temp) :  temporary(temp)
{
	param.size[0] = x;
	param.size[1] = y;
	param.size[2] = z;
	param.rank = 3;
	init_data();
}

TensorCL::TensorCL(int x, int y, int z, int w, bool temp) : temporary(temp)
{
	param.size[0] = x;
	param.size[1] = y;
	param.size[2] = z;
	param.size[3] = w;
	param.rank = 4;
	init_data();
}

TensorCL::TensorCL(cl_tensor p, bool temp) : temporary(temp), param(p)
{
	init_data();
}

TensorCL& TensorCL::operator=(TensorCL & X)
{
	param = X.param;
	//if array is temporary - then just copy the CL pointer
	if (X.isTemporary())
	{
		data = X.data;
		temporary = true;
	}
	else
	{
		init_data();
		clEnqueueCopyBuffer(CL->queue(), data, X.data, 0, 0, param.length * sizeof(float), NULL, NULL, NULL);
		CL->queue.flush();
	}
	
	return *this;
}

TensorCL::TensorCL(TensorCL & X)
{
	*this = X;
}

TensorCL TensorCL::operator+(TensorCL & X)
{
	if (AreTensorsEqual(param, X.param))
	{
		TensorCL C(param, true); //create a temporary array
		add.SetRange(CL->group_size[0], 1, param.length, 1);
		add.SetArg(0, C.data); //result
		add.SetArg(1, data);
		add.SetArg(2, X.data);
		add.SetArg(3, param);
		add.SetArg(4, 1.f);
		add.SetArg(5, 1.f);
		add.RFlush();

		if (X.isTemporary())
		{
			X.release(); //clear memory after temporary array has been used
		}

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
		TensorCL C(param, true); //create a temporary array
		add.SetRange(CL->group_size[0], 1, param.length, 1);
		add.SetArg(0, C.data); //result
		add.SetArg(1, data);
		add.SetArg(2, X.data);
		add.SetArg(3, param);
		add.SetArg(4, 1.f);
		add.SetArg(5, -1.f);
		add.RFlush();

		if (X.isTemporary())
		{
			X.release(); //clear memory after temporary array has been used
		}

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
		TensorCL C(param, true); //create a temporary array
		mul.SetRange(CL->group_size[0], 1, param.length, 1);
		mul.SetArg(0, C.data); //result
		mul.SetArg(1, data);
		mul.SetArg(2, X.data);
		mul.SetArg(3, param);
		mul.SetArg(4, 1.f);
		mul.SetArg(5, 1.f);
		mul.RFlush();

		if (X.isTemporary())
		{
			X.release(); //clear memory after temporary array has been used
		}

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
		TensorCL C(param, true); //create a temporary array
		mul.SetRange(CL->group_size[0], 1, param.length, 1);
		mul.SetArg(0, C.data); //result
		mul.SetArg(1, data);
		mul.SetArg(2, X.data);
		mul.SetArg(3, param);
		mul.SetArg(4, 1.f);
		mul.SetArg(5, -1.f);
		mul.RFlush();

		if (X.isTemporary())
		{
			X.release(); //clear memory after temporary array has been used
		}

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

TensorCL TensorCL::operator*(float x)
{
	return MAD(x, 0.f);
}

TensorCL TensorCL::operator/(float x)
{
	return MAD(1.f/x, 0.f);
}

TensorCL TensorCL::dot(TensorCL & X)
{
	if (AreTensorsCompatible(param, X.param))
	{
		TensorCL C(TensorDotResult(param, X.param), true); //create a temporary array

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

		if (X.isTemporary())
		{
			X.release(); //clear memory after temporary array has been used
		}

		return C;
	}
	else
	{
		ERROR_MSG("Incompatible tensors");
	}
	return *this;
}

void TensorUseOpenCL(OpenCL *cl)
{
	CL = cl;
	add.Initialize("tensor_add", CL, CL->group_size[0], 1, 1, 1);
	mul.Initialize("tensor_mul", CL, CL->group_size[0], 1, 1, 1);
	mad.Initialize("tensor_mad", CL, CL->group_size[0], 1, 1, 1);
	m_dot.Initialize("tensor_dot_product", CL, TS, TS, 1, 1);
}

TensorCL TensorCL::MAD(float a, float b)
{
	TensorCL C(param, true); //create a temporary array
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
