#include "CL_TENSOR.h"

bool Tensor::isTemporary()
{
	return temporary;
}

void Tensor::release()
{
	clReleaseMemObject(data);
}

Tensor::~Tensor()
{
	if (!temporary)
	{
		release();
	}
}

void Tensor::init_data()
{
	length =1;
	for (int i = 0; i < rank; i++)
	{
		length *= size[i];
	}
	cl_int status;
	cl_mem data = clCreateBuffer(CL->default_context(), CL_MEM_READ_WRITE, length * sizeof(float), 0, &status);
}

Tensor::Tensor(int r, vector<int> s, bool temp = false) : rank(r), size(s), temporary(temp)
{
	init_data();
}

Tensor::Tensor(int x, bool temp = false) : rank(2), temporary(temp)
{
	size.push_back(x);
	size.push_back(1);
	init_data();
}

Tensor::Tensor(int x, int y, bool temp = false) : rank(2), temporary(temp)
{
	size.push_back(x);
	size.push_back(y);
	init_data();
}

Tensor::Tensor(int x, int y, int z, bool temp = false) : rank(3), temporary(temp)
{
	size.push_back(x);
	size.push_back(y);
	size.push_back(z);
	init_data();
}

Tensor& Tensor::operator=(Tensor & X)
{
	size = X.size;
	rank = X.rank;
	length = X.length;
	//if array is temporary - then just copy the CL pointer
	if (X.isTemporary())
	{
		data = X.data;
	}
	else
	{
		init_data();
		clEnqueueCopyBuffer(CL->queue(), data, X.data, 0, 0, length * sizeof(float), NULL, NULL, NULL);
		CL->queue.flush();
	}
	
	return *this;
}

void Tensor::TensorUseOpenCL(OpenCL *cl)
{
	CL = cl;
}