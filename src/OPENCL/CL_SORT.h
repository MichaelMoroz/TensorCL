

//sort the matrix columns by the first row elements
class BitonicMatrixSort
{
public:
	int M, N;
	int Np2;
    OpenCL* CL;
	cl::Buffer bufpow2;
    CLFunction SORT;

    BitonicMatrixSort()
    {

    }

	void initialize(OpenCL *cl, CLMatrix& A)
	{
		M = A.M;
		N = A.N;
		Np2 = pow(2, ceil(log2(N)));
		CL = cl;
		bufpow2 = cl::Buffer(CL->default_context, CL_MEM_READ_WRITE, M * Np2 * sizeof(float));
		SORT.Initialize("ParallelBitonic", CL, 1, 8, 1, Np2/2);
		SORT.SetArg(0, A.buffer_matrix);
		SORT.SetArg(1, bufpow2);
		SORT.SetArg(2, M);
		SORT.SetArg(3, N);
		SORT.SetArg(4, Np2);
	}

	void SetSortRange(int SR)
	{
		N = SR;
		Np2 = pow(2, ceil(log2(N)));
		SORT.SetRange(1, 8, 1, Np2 / 2);
		SORT.SetArg(3, N);
		SORT.SetArg(4, Np2);
	}

    void sort()
    {
        for (int length=1; length<Np2; length<<=1)
        {
            int inc = length;
            while (inc > 0)
            {
				SORT.SetArg(5, inc);
				SORT.SetArg(6, length << 1);

				if (length == 1 && inc == 1) //first iteration
				{
					SORT.SetArg(7, -1); //copy data to buffer
				}
				else if(length == Np2>>1 && inc == 1) //last iteration
				{
					SORT.SetArg(7, 1); //copy data from buffer
				}
				else
				{
					SORT.SetArg(7, 0); //use only the buffer
				}

				SORT.RFlush();

                inc >>= 1;
            }
        }
    }

};
