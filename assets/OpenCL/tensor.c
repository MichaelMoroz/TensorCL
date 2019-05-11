#define MAX_DIM 8
#define TS 8

#pragma pack(push, r1, 1)
typedef struct
{
	int size[MAX_DIM];
	int rank;
	int length;
} cl_tensor;
#pragma pack(pop, r1)

#pragma pack(push, r1, 1)
typedef struct
{
	int index[MAX_DIM];
} cl_index;
#pragma pack(pop, r1)


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

int GetDelta(cl_tensor info, int N)
{
	int sh = 1;
	for (int i = 0; i < N; i++)
	{
		sh *= info.size[i];
	}
	return sh;
}

bool is_diag(cl_index indx, cl_tensor info)
{
	bool diag = true;
	for (int i = 0; i < info.rank-1; i++)
	{
		diag = diag && (indx.index[i] == indx.index[i+1]);
	}
	return diag;
}

cl_index get_index(int id, cl_tensor info)
{
	cl_index idx;
	int sh = 1;
	for (int i = 0; i < MAX_DIM; i++)
	{
		idx.index[i] = id/sh - info.size[i] * (id /(info.size[i] * sh));
		sh *= info.size[i];
	}
	return idx;
}

cl_index transpose(cl_index T, int a, int b)
{
	int buf = T.index[a];
	T.index[a] = T.index[b];
	T.index[b] = buf;
	return T;
}

__kernel void tensor_add( __global float* C,
					const __global float* A,
					const __global float* B,
					const cl_tensor Cdata,
					const float a,
					const float b)
{
	const int i = get_global_id(0);
	if(i < Cdata.length)
		C[i] = A[i] * a + B[i] * b;
}

__kernel void tensor_mul( __global float* C,
					const __global float* A,
					const __global float* B,
					const cl_tensor Cdata,
					const float a,
					const float b)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = pow(A[i], a)*pow(B[i], b);
}

__kernel void tensor_less_m(__global float* C,
	const __global float* A,
	const __global float* B,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = A[i] < B[i];
}

__kernel void tensor_more_m(__global float* C,
	const __global float* A,
	const __global float* B,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = A[i] > B[i];
}

__kernel void tensor_less_n(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const float a)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = A[i] < a;
}

__kernel void tensor_more_n(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const float a)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = A[i] > a;
}

__kernel void tensor_if(__global float* C,
	const __global float* COND,
	const __global float* TRUE,
	const __global float* FALSE,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = ((bool)COND[i])?TRUE[i]:FALSE[i];
}

__kernel void tensor_index(__global float* C,
					const cl_tensor Cdata,
					int dim)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = get_index(i, Cdata).index[dim];
}

__kernel void tensor_mad(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const float a,
	const float b)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = A[i] * a + b;
}

__kernel void tensor_sin(__global float* C,
	const __global float* A,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = sin(A[i]);
}

__kernel void tensor_cos(__global float* C,
	const __global float* A,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = cos(A[i]);
}

__kernel void tensor_tan(__global float* C,
	const __global float* A,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = tan(A[i]);
}

__kernel void tensor_pow(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const float power)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = pow(A[i],power);
}

__kernel void tensor_exp(__global float* C,
	const __global float* A,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = exp(A[i]);
}

__kernel void tensor_log(__global float* C,
	const __global float* A,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = log(A[i]);
}

__kernel void tensor_tanh(__global float* C,
	const __global float* A,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = tanh(A[i]);
}

__kernel void tensor_random(__global float* C,
	const cl_tensor Cdata,
	const int seed)
{
	const int i = get_global_id(0);
	int tseed = 102000*(seed + i);
	if (i < Cdata.length)
		C[i] = sfrand(&tseed);
}

__kernel void tensor_min(__global float* C,
	const __global float* A,
	const __global float* B,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = min(A[i], B[i]);
}

__kernel void tensor_max(__global float* C,
	const __global float* A,
	const __global float* B,
	const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = max(A[i], B[i]);
}

__kernel void tensor_min_f(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const float b)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = min(A[i], b);
}

__kernel void tensor_max_f(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const float b)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = max(A[i], b);
}

__kernel void tensor_transpose(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const cl_tensor Adata,
	const int dim_a,
	const int dim_b )
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
	{
		int id = ID(transpose(get_index(i, Cdata), dim_a, dim_b), Adata);
		if(id < Adata.length)
			C[i] = A[id];
	}
}

__kernel void tensor_cut(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const cl_tensor Adata,
	const int from,
	const int to )
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
	{
		cl_index idx = get_index(i, Cdata);
		idx.index[Adata.rank-1] += from; 
		int id = ID(idx, Adata);
		if(id < Adata.length)
			C[i] = A[id];
	}
}


__kernel void tensor_diag(__global float* C,
	const cl_tensor Cdata,
	const float x,
	const float y )
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = is_diag(get_index(i, Cdata), Cdata)?x:y;
}

__kernel void tensor_repeat(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const cl_tensor Adata,
	const int n)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
	{
		int id = ID(get_index(i, Cdata), Adata);
		if(id < Adata.length)
			C[i] = A[id];
	}
}

__kernel void tensor_sum(__global float* C,
	const __global float* A,
	const cl_tensor Cdata,
	const cl_tensor Adata)
{
	const int i = get_global_id(0);
	float sum = 0.f;

	int did = GetDelta(Adata, Adata.rank-1);
	for (int j = 0; j < Adata.size[Adata.rank-1]; j++)
	{
		if(i + j*did < Adata.length)
			sum += A[i + j*did];
	}
	
	if (i < Cdata.length)
		C[i] = sum;
}

#define NAIVE_DOT false

__kernel void tensor_dot_product( __global float* C,
							__global float* A,
							__global float* B,
							const cl_tensor Cdata,
							const cl_tensor Adata,
							const cl_tensor Bdata,
							const int shiftA,
							const int shiftB,
							const int shiftC)
{
	const int M = Adata.size[0];
	const int K = Adata.size[1];
	const int N = Bdata.size[1];

	// Thread identifiers
	const int row = get_local_id(0); // Local row ID (max: TS)
	const int col = get_local_id(1); // Local col ID (max: TS)

	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)

	// Compute a single element (loop over K)
	float acc = 0.0f;
	
	#if NAIVE_DOT
		for (int k=0; k<K; k++) 
		{
			acc += A[k*M + globalRow + shiftA] * B[globalCol*K + k + shiftB];
		}
	#else
	    __local float Asub[TS][TS];
		__local float Bsub[TS][TS];
		
		// Loop over all tiles
		const int numTiles = K/TS + 1;

		for (int t = 0; t < numTiles; t++) {

			// Load one tile of A and B into local memory
			const int tiledRow = TS * t + row;
			const int tiledCol = TS * t + col;
			
			int idA = tiledCol*M + globalRow + shiftA;
			int idB	= globalCol*K + tiledRow + shiftB;
			Asub[col][row] = (tiledCol < K && idA < Adata.length) ? (A[idA]) : (0.f);
			Bsub[col][row] = (tiledRow < K && idB < Bdata.length) ? (B[idB]) : (0.f);

			// Synchronise to make sure the tile is loaded
			barrier(CLK_LOCAL_MEM_FENCE);

			// Perform the computation for a single tile
			for (int k = 0; k < TS; k++) {
				acc += Asub[k][row] * Bsub[col][k];
			}

			// Synchronise before loading the next tile
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
	#endif
		
	// Store the result
	if (globalRow < M && globalCol < N)
		C[globalCol*M + globalRow + shiftC] = acc;
}