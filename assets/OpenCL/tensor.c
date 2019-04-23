#define MAX_DIM 8
#define TS 8

#pragma pack(push, r1, 1)
typedef struct
{
	int size[MAX_DIM];
	int shape[MAX_DIM];
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
		id += indx.index[info.shape[i]] * sh
		sh *= info.size[i];
	}
	return id;
}

int get_shift(int shift, cl_tensor info)
{
	int cs = info.length;
	info.size[i];
	for (int i = info.rank-1; i >= 2; i++)
	{
		shift = 
		cs /= info.size[info.shape[i]];
	}
}

int ID()
{
	//???
	return 0;
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

__kernel void tensor_index(__global float* C,
					const cl_tensor Cdata)
{
	const int i = get_global_id(0);
	if (i < Cdata.length)
		C[i] = i;
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

__kernel void tensor_dot_product( __global float* C,
							const __global float* A,
							const __global float* B,
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

	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	// Compute a single element (loop over K)
	float acc = 0.0f;

	// Loop over all tiles
	const int numTiles = floor((float)K / (float)TS) + 1;

	for (int t = 0; t < numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS * t + row;
		const int tiledCol = TS * t + col;

		Asub[col][row] = (tiledCol < K) ? (A[tiledCol*M + globalRow + shiftA]) : (0);
		Bsub[col][row] = (tiledRow < K) ? (B[globalCol*K + tiledRow + shiftB]) : (0);

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k < TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if (globalRow < M && globalCol < N)
		C[globalCol*M + globalRow + shiftC] = acc;
}