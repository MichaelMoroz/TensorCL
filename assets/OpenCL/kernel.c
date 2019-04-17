
//sudo-random generator
float random(float *seed)
{
	*seed = 1510546 * (*(int*)seed);
	*seed = ((int)(*seed)) >> 9 + 143545;
	return ((((int)(*seed)) >> 3) % 32000)/(float)32000;
}

float randd(float a, float b, float *seed)
{
	return (b - a)*random(seed) + a;
}

#define M_PI 3.14159265

//box-muller gausian random numbers
float randomn(float sigma, float *seed)
{
	return sigma * sqrt(-2 * log(random(seed) + 1e-8))*cos(2 * 3.14159*random(seed));
}

float abz(float x)
{
	return (x > 0) ? x : -x;
}

float abc(float x)
{
	return (x > 0) ? x : 0;
}


float powint(float x, int p)
{
	float o = 1;
	for (int i = 0; i < p; i++)
	{
		o *= x;
	}
	return o;
}

float cdf(float d, float cd)
{
	return (d > cd) ? 0 : (0.5*(native_cos(M_PI*d / cd) + 1));
}

float symfun(float d, float4 D, float4 dx, float q, float eta, float cdfv, float cd)
{
	float d0 = length(D - dx);
	return q*exp(-pow(0.75*d0/cd,2))*cdfv/32;
}

//the charge of the last electron shell
float QTR(float q, int i)
{
	if (i == 0)
	{
		return q;
	}
	else
	{
		return q*sin(2 * i * q / (18/*atom group size*/));
	}
}


//create the atom lookup block dependance matrix
__kernel void LookupUpdate1(__global float* Lookup1,
	const __global float* XYZ,
	const int N, const int BS, const float CS, const int S)
{
	const int AT = get_global_id(1); // AtomID

	int i = XYZ[AT * 3] / CS + BS*0.5f, j = XYZ[AT * 3 + 1] / CS + BS*0.5f, k = XYZ[AT * 3 + 2] / CS + BS*0.5f; //this atom block

	Lookup1[AT * 2 + 0] = BS*BS*S*i + BS*S*j + S*k;
	Lookup1[AT * 2 + 1] = AT;
}

int cutt(int id, int N)
{
	return (id < 0) ? (0) : ((id < N) ? id : (N - 1));
}



//Update the lookup blocks after sorting the previous matrix
__kernel void LookupUpdate2(__global float* Lookup,
	const __global float* Lookup1,
	const int N, const int BS, const float CS, const int S, const int Iteration)
{
	const int AT = get_global_id(1); //lookup1 id
	int locid, prvid;


	bool copy = 0;
	int i = 0;

	locid = Lookup1[AT * 2 + 0];
	prvid = Lookup1[cutt(AT - 1, N) * 2 + 0];

	//if first of this block then update
	if (AT == 0 || locid != prvid)
	{
		copy = 1;
	}

	//if first update of this block, otherwise do nothing(to secure the access of the lookup table memory)
	if (copy)
	{
		int p = AT;
		int tid = locid;
		bool ok = 1;
		//set that this block was updated this iteration
		Lookup[tid] = Iteration;

		//if this block is the same as the previous
		while (i < S-3)
		{
			if (ok && p < N)
			{
				//save this atom id to the lookup table
				Lookup[locid + i +1] = Lookup1[p * 2 + 1];

				if (p + 1 < N)
				{
					if (Lookup1[(p + 1) * 2 + 0] != locid)
					{
						ok = 0;
					}
				}
			}
			else
			{
				Lookup[tid + i + 1] = -1;
			}
			p++;
			i++;
		}
	}
}

float4 GenPos(float sd)
{
	float seed = sd;
	float4 a = (float4)(2, 0, 0, 0);
	while (length(a) > 1)
	{
		a.x = random(&seed);
		a.y = random(&seed);
		a.z = random(&seed);
	}
	return a;
}

float cn(float x)
{
	x = (x > 0) ? x : -x;
	return (x < 1) ? (1 - x) : 0;
}


float4 SymmetryF(float4 D, float k)
{
	float d = length(D);
	return D*(exp(-d*k) / d);
}

float4 SymmetryF_der(float4 D, float k)
{
	float d = length(D);
	return -D*exp(-d*k);
}


float dc(float d, float dcut, int n, int i)
{
	//return abc(1 - (float)n*abz((d / dcut - (float)i / (float)n))); //version 1 2.04
	//return exp(-pow((float)n*(d / dcut - (float)i / (float)n),2)); //version 2 2.01
	//return exp(-pow((float)n*(d/dcut - (float)i/(float)n),2))*cdf(d, dcut); //version 3 2.01
	return exp(-pow((float)0.5*n*(d / dcut - (float)i / (float)n), 2))*cdf(d, dcut);
}

float c2m(int a)
{
	if (a == 8)
	{
		return 15.999f;
	}
	else if (a == 30)
	{
		return 65.38f;
	}
	else
	{
		return 1;
	}
}


__kernel void StructOptim(__global float* XYZ,
	const __global float* Force,
	const int N, const float ds)
{
	const int j = get_global_id(1); // AtomID
	float4 F = (float4)(Force[j * 3], Force[j * 3 + 1], Force[j * 3 + 2], 0.f);
	float seed = 232 * F.x + 0.5f * F.y * F.z;

	F.x += randd(-0.15, 0.15, &seed);
	F.y += randd(-0.15, 0.15, &seed);
	F.z += randd(-0.15, 0.15, &seed);


	XYZ[j * 3] += ds*F.x;
	XYZ[j * 3 + 1] += ds*F.y;
	XYZ[j * 3 + 2] += ds*F.z;
}

__kernel void SimulIter(__global float* XYZ,
	__global float* V,
	const __global float* F,
	const __global float* C,
	const int N, const float dts, const float CS)
{
	//dt is in femtoseconds
	const int AT = get_global_id(1); // AtomID
	int i = 0;
	float m = c2m((int)C[AT]); //atom mass in amu
	float4 v = (float4)(V[AT * 3], V[AT * 3 + 1], V[AT * 3 + 2], 0);
	float4 f = (float4)(F[AT * 3], F[AT * 3 + 1], F[AT * 3 + 2], 0);
	float4 x = (float4)(XYZ[AT * 3], XYZ[AT * 3 + 1], XYZ[AT * 3 + 2], 0);
	//the coefficient is to convert eV/(Angstrom*amu) to acceleration Angstrom/fs^2
	v = v - f*0.5*0.001f; /*- 0.00001*dt*103.645f*f*/; //velocity cahange (Angstrom/femtosecond)
	float dt = 0.5f;
	if (x.x + dt*v.x < CS)
	{
		V[AT * 3 + i] = v.x;
		XYZ[AT * 3 + i] = x.x + dt*v.x;
	}
	else
	{
		V[AT * 3 + i] = -v.x;
		XYZ[AT * 3 + i] = x.x - dt*v.x;
	}

	i = 1;
	if (x.y + dt*v.y < CS)
	{
		V[AT * 3 + i] = v.y;
		XYZ[AT * 3 + i] = x.y + dt*v.y;
	}
	else
	{
		V[AT * 3 + i] = -v.y;
		XYZ[AT * 3 + i] = x.y - dt*v.y;
	}

	i = 2;
	if (x.z + dt*v.z < CS)
	{
		V[AT * 3 + i] = v.z;
		XYZ[AT * 3 + i] = x.z + dt*v.z;
	}
	else
	{
		V[AT * 3 + i] = -v.z;
		XYZ[AT * 3 + i] = x.z - dt*v.z;
	}

}

__kernel void SymmetryCalc(const __global float* Lookup,
	const __global float* XYZ,
	const __global float* Q,
	const __global float* Sphere,
	__global float* SymmetryFunc,
	const int N, const int BS, const float CS, const int S,
	const float cut_d, const int Iteration,
	const int SphereN, const int AtomTypeN)
{
	const int AT = get_global_id(1); // AtomID

	if (AT < N)
	{
		float4 X = (float4)(XYZ[AT * 3], XYZ[AT * 3 + 1], XYZ[AT * 3 + 2], 0);//our cental atom

		float q0 = Q[AT]; //our atom type

		//lookup center position
		float4 CC = (float4)(BS*0.5f + X.x / CS, BS*0.5f + X.y / CS, BS*0.5f + X.z / CS, 0);

		float DD = cut_d / CS + 1;

		//neighbore buffer 256 max
		float4 ID_XYZ[256];

		//neighbore number
		int ii = 0;

		///search through the lookup table for the atom neighbores
		//x
		for (int i = round((float)(CC.x - DD)); i <= CC.x + DD; i++)
		{
			//y
			for (int j = round((float)(CC.y - DD)); j <= CC.y + DD; j++)
			{
				//z
				for (int k = round((float)(CC.z - DD)); k <= CC.z + DD; k++)
				{
					if (i >= 0 && i < BS && j >= 0 && j < BS && k >= 0 && k < BS)
					{
						if (Lookup[((int)(BS*BS*S*i + BS*S*j + S*k))] == Iteration)
						{
							//x = CC.x;
							//search over all atoms inside this block
							for (int t = 1; t < S; t++)
							{
								//atom id
								int id = Lookup[((int)(BS*BS*S*i + BS*S*j + S*k + t))];
								//if no more atoms or old lookup block then break loop
								if (id == -1)
								{
									break;
								}
								//calculate the symmetry functions if not the same atom
								else if (id != AT)
								{	
									float4 Y = (float4)(XYZ[id * 3], XYZ[id * 3 + 1], XYZ[id * 3 + 2], 0); //neighbore atom coordinate
									float4 D = X - Y; //distance radius vector
									float d = length(D); //scalar distance
									 //if inside the cut sphere
									if (d < cut_d)
									{
										bool itsok = 1;
										
										if (itsok)
										{
											ID_XYZ[ii].x = D.x;
											ID_XYZ[ii].y = D.y;
											ID_XYZ[ii].z = D.z;
											ID_XYZ[ii].w = Q[id];
											ii++;
										}
									}
								}
							}
						}
					}
					
				}
			}
		}

		//sphere point buffer
		float4 SPH[128];
		//copy to buffer
		for (int i = 0; i < SphereN; i++)
		{
			SPH[i].x = Sphere[i * 3 + 0];
			SPH[i].y = Sphere[i * 3 + 1];
			SPH[i].z = Sphere[i * 3 + 2];
			SPH[i].w = 0;
		}
		const int snf = 12;
		float a[12];
		
		//calculate the symmetry functions
		for (int s = 0; s < SphereN; s++)
		{
			float ds = length(SPH[s]);
			for (int t = 0; t < AtomTypeN; t++)
			{
				int pp = (q0 == t) ? 1 : -1;
				float inv_d = 0, amount = 0;
				for (int ai = 0; ai < snf;ai++ )
				{
					a[ai] = 0;
				}
				float4 C = (float4)(0,0,0,0);
				for (int i = 0; i < ii; i++)
				{
					if (ID_XYZ[i].w == t)
					{
						//radius vector
						float4 D = (float4)(ID_XYZ[i].x, ID_XYZ[i].y, ID_XYZ[i].z, 0);
						float d = length(D);
						float ang = acos((D.x*SPH[s].x+D.y*SPH[s].y+D.z*SPH[s].z)/(d*ds));
						float k = exp(-pow(ang/(M_PI/sqrt((float)SphereN)),2));

						//C += D*k;
						for (int ai = 0; ai < snf; ai++)
						{
							a[ai] += k*dc(d,cut_d,snf,ai);
						}
						
					}
				}
			
				///save this point and type
				for (int ai = 0; ai < snf;ai++)
				{
					SymmetryFunc[AT * SphereN * AtomTypeN * snf + s * AtomTypeN * snf + t * snf + ai] = pp*a[ai];
				}
			}
		
		}
		
	}
}



__kernel void AtomPairCalc(const __global float* Lookup,
	const __global float* XYZ,
	const __global float* Q,
	__global float* AtomPairs,
	__global float* SumMat,
	const int N, const int BS, const float CS, const int S,
	const float cut_d, const int Iteration, const int AtomTypeN)
{
	const int AT = get_global_id(1); // AtomID

	if (AT < N)
	{
		float4 X = (float4)(XYZ[AT * 3], XYZ[AT * 3 + 1], XYZ[AT * 3 + 2], 0);//our cental atom

		float q0 = Q[AT]; //our atom type

						  //lookup center position
		float4 CC = (float4)(BS*0.5f + X.x / CS, BS*0.5f + X.y / CS, BS*0.5f + X.z / CS, 0);

		float DD = cut_d / CS + 1;

		//neighbore buffer 256 max
		float4 ID_XYZ[256];

		//neighbore number
		int ii = 0, ij = 0;

		///search through the lookup table for the atom neighbores
		//x
		for (int i = round((float)(CC.x - DD)); i <= CC.x + DD; i++)
		{
			//y
			for (int j = round((float)(CC.y - DD)); j <= CC.y + DD; j++)
			{
				//z
				for (int k = round((float)(CC.z - DD)); k <= CC.z + DD; k++)
				{
					if (i >= 0 && i < BS && j >= 0 && j < BS && k >= 0 && k < BS)
					{
						if (Lookup[((int)(BS*BS*S*i + BS*S*j + S*k))] == Iteration)
						{
							//x = CC.x;
							//search over all atoms inside this block
							for (int t = 1; t < S; t++)
							{
								//atom id
								int id = Lookup[((int)(BS*BS*S*i + BS*S*j + S*k + t))];
								//if no more atoms or old lookup block then break loop
								if (id == -1)
								{
									break;
								}
								//calculate the symmetry functions if not the same atom
								else if (id != AT)
								{
									float4 Y = (float4)(XYZ[id * 3], XYZ[id * 3 + 1], XYZ[id * 3 + 2], 0); //neighbore atom coordinate
									float4 D = X - Y; //distance radius vector
									float d = length(D); //scalar distance
														 //if inside the cut sphere
									if (d < cut_d)
									{
										bool itsok = 1;

										if (itsok)
										{
											ID_XYZ[ii].x = D.x;
											ID_XYZ[ii].y = D.y;
											ID_XYZ[ii].z = D.z;
											ID_XYZ[ii].w = Q[id];
											ii++;
										}
									}
								}
								ij++;
							}
						}
					}

				}
			}
		}

		//loop  over all neighbore atoms
		for (int i = 0; i < ii; i++)
		{
			
			//radius vector
			float4 D = (float4)(ID_XYZ[i].x, ID_XYZ[i].y, ID_XYZ[i].z, 0);
			float d = length(D);
			D = D / d;
			float mm = 1;
			if (ID_XYZ[i].w == q0)
			{
				mm = -1;
			}

			///save the pair
			//normalized radius vector to the second atom
			AtomPairs[AT * (3 + AtomTypeN) * S + i*(3 + AtomTypeN) + 0] = D.x;
			AtomPairs[AT * (3 + AtomTypeN) * S + i*(3 + AtomTypeN) + 1] = D.y;
			AtomPairs[AT * (3 + AtomTypeN) * S + i*(3 + AtomTypeN) + 2] = D.z;

			//interaction parameters
			for (int p = 0; p < AtomTypeN; p++)
			{
				float a = 0;
				if (p == ID_XYZ[i].w)
				{
					a = mm / d;
				}
				AtomPairs[AT * (3 + AtomTypeN) * S + i*(3 + AtomTypeN) + 3 + p] = a;
			}	
		}

		SumMat[AT] = ii;

	}
}

__kernel void weight_cut(const float a,
	__global float* A,
	const __global float* data)
{
	int M = data[0];
	int N = data[1];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		if (abz(A[j*M + i]) < a)
		{
			A[j*M + i] = 0;
		}
	}
}

__kernel void Randomize(const float a, const float b, const float seed1,
	__global float* A,
	const __global float* data)
{
	const int M = data[0];
	const int N = data[1];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		float seed = seed1*(i*N + j);

		A[j*M + i] = random(&seed)*(b - a) + a;
	}
}


__kernel void set_kernel(__global float* A, const float x,
	const __global float* data)
{
	int M = data[0];
	int N = data[1];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID

	if (i < M && j < N)
	{
		A[j*M + i] = x;
	}
}

__kernel void error_convert(
	__global float* EN,//new error
	const __global float* EO,//old error
	const __global float* data)
{
	const int M = data[0];//rows NEURONS
	const int N = data[1];//cols SAMPLES
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		EN[j*M + i]/*size = Neurons*Samples */ = EO[i*N + j]/*size = Samples*(Neurons+1) */;
	}
}


__kernel void generate_input_set(
	__global float* A,//input set
	const __global float* B,//coord of the constraint points
	const __global float* C,//coordinates of the random set
	const __global float* data,
	const float rand, const float dimensions, const float components,//optimization parameters
	const float order, const float constr_num //train iteration
)
{
	int M = data[0]; //dimesions
	int N = data[1]; //samples
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	float seed1 = rand+j;
	float seed2 = rand+j+i*23;
	if (10*random(&seed1) < 3)//add 30% of constraint points
	{
		int k = (int)(constr_num * random(&seed1));
		A[j*M + i] = B[k*M + i];
	}
	else
	{
		A[j*M + i] = C[j*M + i] + randomn(0.5, &seed2);
	}
	
}



__kernel void generate_derivative_set(
	const __global float* B,//coord of the input points
	__global float* A,//input set
	const __global float* data,
	const float rand, const float dimensions, const float components,//optimization parameters
	const float order, const float constr_num //train iteration
)
{
	int M = data[0]; //dimesions
	int N = data[1]; //samples
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	int s = 1 + dimensions * 2;
	float d = 0.001;
	for (int k = 0; k < s; k++)
	{
		if (k == 0) // the central point
		{
			A[(s*N - k*s - j)*M + i] = B[j*M + i];
		}
		else // derivative points
		{
			if ((k+1)/2 == i) //if the correct dimension add or substract d
			{
				if (k % 2 == 0)
				{
					A[(s*N - (k + 1)*N + j)*M + i] = B[j*M + i] + d;
				}
				else
				{
					A[(s*N - (k + 1)*N + j)*M + i] = B[j*M + i] - d;
				}
			} 
			else // for all other dimensions
			{
				A[(s*N - (k + 1)*N + j)*M + i] = B[j*M + i];
			}
		}
	}
}


__kernel void generate_train_set(
	__global float* A,//train set
	const __global float* B,//coord of the input set
	const __global float* C,//order 0 constraints
	const __global float* D,//order 1 constraints
	const __global float* E,//field values at input points
	const __global float* F,//coordinates of the random set
	const __global float* data,
	const float rand, const float dimensions, const float components,//optimization parameters
	const float order, const float constr_num //train iteration
)
{
	const int M = data[0]; //dimesions
	const int N = data[1]; //samples
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		float seed1 = rand + j;
		float seed2 = rand + j + i * 23;
		if (10 * random(&seed1) < 3) //add 30% of constraint points
		{
			int k = (int)(constr_num * random(&seed1));
			A[j*M + i] = B[k*M + i];
		}
		else
		{
			A[j*M + i] = C[j*M + i] + randomn(0.5, &seed2);
		}
	}

}

float limit(float x, float l)
{
	return l*tanh(x/l);
}

//gradient descent augmented with ADAM + neterov
__kernel void ADAM(
	__global float* A,//weights
	const __global float* B,//gradient
	__global float* C,//gradient rolling average
    __global float* D,//gradient square rolling average
	__global float* E,//gradient square rolling average
	const __global float* data,
	const __global float* data1
	)
{
	
	const int M = data[0]; //Neurons Layer i+1
	const int N = data[1]; //Neurons Layer i + bias neuron
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID

	if (i < M && j < N)
	{
		const float alpha = data1[0];
		const float beta1 = data1[1];
		const float beta2 = data1[2];
		const int IT = data1[3];
		float G = B[i*N + j]/2000;
		float Mg = beta1*C[j*M + i] + G*(1 - beta1);
		float Mv = beta2*D[j*M + i] + G*G*(1 - beta2); // adam


		C[j*M + i] = Mg;
		D[j*M + i] = Mv;

		if (IT < 15)
		{
			Mg /= (1 - powint(beta1, IT + 1));
			Mv /= (1 - powint(beta2, IT + 1));
		}
		//Weight update
		A[j*M + i] -= alpha * G/*(beta1*Mg + (1 - beta1)*G) / (sqrt(Mv) + 1e-8)*/;
	}
}

float sgn1(float x)
{
	return (x >= 0) ? (1) : (-1);
}

//activation function kernel
float activ(float x, int id)
{
	const float alpha = 0.05;
	const float xm = 2.4;
	switch (id)
	{
		/*usual stuff*/
	//linear
	case -1:
		return x;
	//sigmoid
	case 0: 
		return 1 / (1 + native_exp(-x));
	//tanh
	case 1:
		return tanh(x);
	//ReLU
	case 2:
		return (x > 0) ? x : 0.f;
	//Leaky ReLU
	case 3:
		return ((x > 0) ? x : alpha*x);
	//ELU
	case 4:
		return (x > 0) ? x : alpha*(exp(x)-1);
		/*unusual stuff*/
	//hyperbolic sine
	case 5:
		return sinh(x);
	//hyperbolic sine inverse
	case 6:
		return native_log(x + native_sqrt(x*x + 1));
	//sine
	case 7:
		return native_sin(x);
	//inverse
	case 8:
		return 1 / (abz(x) + 1);
	//square
	case 9:
		return x*x;
	//modified sqare root
	case 10:
		return sgn1(x)*(native_sqrt(abz(x) + 0.1) - native_sqrt(0.1));
	//cut exponent
	case 11:
		return (x < -xm) ? (exp(-xm)*(1 + x + xm)) : ((x > xm) ? (exp(xm)*(1 + x - xm)) : (exp(x)));
	//cut logarithm
	case 12:
		return (x < native_exp(-xm)) ? (-xm + exp(xm)*x - 1) : ((x > exp(xm)) ? (xm + exp(-xm)*x - 1) : (log(x)));
	//tanh + alpha*x
	case 13:
		return tanh(x) + alpha*x;
	//gauss
	case 14:
		return native_exp(-x*x);
	}
}

//activation function derivatives
//neuron output, neuron input, activation id
float activ_deriv(float a, float x, int id)
{
	const float alpha = 0.05;
	const float xm = 2.4;
	switch (id)
	{
		/*usual stuff*/
		//linear
	case -1:
		return 1;
		//sigmoid
	case 0:
		return a*(1-a);
		//tanh
	case 1:
		return 1-a*a;
		//ReLU
	case 2:
		return (a > 0) ? 1 : 0;
		//Leaky ReLU
	case 3:
		return (a > 0) ? 1 : alpha;
		//ELU
	case 4:
		return (a > 0) ? 1 : alpha+a;
		/*unusual stuff*/
		//hyperbolic sine
	case 5:
		return cosh(x);
		//hyperbolic sine inverse
	case 6:
		return rsqrt(x*x + 1);
		//sine
	case 7:
		return native_cos(x);
		//inverse
	case 8:
		return (x > 0) ? a*a : -a*a;
		//square
	case 9:
		return 2*x;
		//sqare root
	case 10:
		return native_rsqrt(abz(x) + 1);
		//cut exponent
	case 11:
		return (x < -xm) ? (exp(-xm)) : ((x > xm) ? (exp(xm)) : (a));
		//cut logarithm
	case 12:
		return (x < exp(-xm)) ? (exp(xm)) : ((x > exp(xm)) ? (exp(-xm)) : (1/x));
	//tanh + alpha
	case 13:
		a = a - alpha*x;
		return 1 - a*a + alpha;
	//gauss
	case 14:
		return - 2 * x * a;
	}
}

__kernel void activation(__global float* A,
	const __global float* B,//neuron input
	const __global float* A_ID,//neuron activation
	const __global float* data)
{
	int M = data[0];
	int N = data[1];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		A[j*M + i] = activ(B[j*M + i], (int)A_ID[i]);
	}
}


__kernel void FastSum(__global float* Buf,
	const __global float* data,
	const int Level,
	const int K
)
{
	int N = data[0]*data[1];
	const int i = get_global_id(0); // element id
	//sum over a subarray
	float sum = 0;
	for (int j = 0; j < K && (i*(int)powint(K, Level) + j*(int)powint(K, Level - 1) < N); j++)
	{
		sum += Buf[i*(int)powint(K, Level) + j*(int)powint(K, Level - 1)];
	}
	Buf[i*(int)powint(K, Level)] = sum;
}

__kernel void FastSumMod(__global float* Buf,
	const __global float* data,
	const int Level,
	const int K
)
{
	int N = data[0] * data[1];
	const int i = get_global_id(0); // element id
									//sum over a subarray
	float sum = 0;
	for (int j = 0; j < K && (i*(int)powint(K, Level) + j*(int)powint(K, Level - 1) < N); j++)
	{
		sum += abz(Buf[i*(int)powint(K, Level) + j*(int)powint(K, Level - 1)]);
	}
	Buf[i*(int)powint(K, Level)] = sum;
}

__kernel void copy_i(__global float* sub,//in batch
	const __global float* IN,//in samples
	const __global float* data,
	const float sample0)
{
	int M = data[0];
	int N = data[1];//sample dimension
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		sub[j*M + i] = IN[(j + (int)sample0)*M + i];
	}
}

__kernel void copy_o(const __global float* OUTP,//in batch
	 __global float* out,//in samples
	const __global float* data,
	const float sample0)
{
	int M = data[0];
	int N = data[1];//sample dimension
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		out[(j + (int)sample0)*M + i] = OUTP[j*M + i];
	}
}

__kernel void error_calc(__global float* Error,//error
	const __global float* OUT,//NN output
	const __global float* TAR,//NN target
	const int M,
	const int N,
	const int samp)
{
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		Error[j*M + i] = OUT[j*M + i] - TAR[(j + samp)*M + i];
	}
}

__kernel void error_calc_grad(__global float* Error,//error
	const __global float* data,
	const __global float* data1)
{
	int M = data[0];//neuron dimension
	int N = data[1];//sample dimension
	int ii = data1[0];//chosen output neuron
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		//if neuron == chosen
		if (i == ii)
		{
			Error[j*M + i] = 1;
		}
		else
		{
			Error[j*M + i] = 0;
		}
	}
}

__kernel void copy_grad(__global float* GRADIENT, //Grad matrx
	const __global float* error,
	const __global float* error_dat, 
	const __global float* grad_dat, 
	const __global float* data)
{
	int M = error_dat[0]; //inputs
	int N = error_dat[1]; //samples
	int dN = data[0]*data[1]; //grad shift
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		GRADIENT[(j + dN)*M + i] = error[j*M + i];
	}
}

__kernel void delta_calc(const __global float* O_L, //neuron input OUTPUT_L
	__global float* delta,//transposed delta
	const __global float* O, //neuron output OUTPUT
	const __global float* Error, //Error from the previous layer(i.e next one from start)
	const __global float* A_ID, //neuron activations
	const int M, const int N)
{
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		delta[i*N + j] = activ_deriv(O[j*M + i], O_L[j*M + i], (int)A_ID[i])*Error[j*M + i];
	}
}

//copy kernel
__kernel void copy_kernel(__global float* A,
			 const __global float* B,
			 const __global float* data) 
{
	int N = data[1];
	int M = data[0];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		A[j*M + i] = B[j*M + i];
	}
}

__kernel void copy_kernel_input(__global float* A,
	const __global float* B,
	const int M, const int N, const int samp)
{
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		A[j*M + i] = B[(j+samp)*M + i];
	}
}

__kernel void copy_kernel_both(__global float* A,
	const __global float* B,
	const int M, const int N, const int sampa, const int sampb)
{
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		A[(j + sampa)*M + i] = B[(j + sampb)*M + i];
	}
}

__kernel void ckbl(__global float* A,
	const __global float* B,
	const int M, const int N, const int sampa, const int sampb, const int lima, const int limb)
{
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && (j) < N && (j+sampb)<limb)
	{
		A[(j + sampa)*M + i] = B[(j + sampb)*M + i];
	}
}

__kernel void Pooling(const __global float* OUT,
	 __global float* IN,
	const __global float* SUM,
	const int M, const int Nin, const int Nout, const int sumsamp)
{
	const int i = get_global_id(0); // in row ID 
	const int j = get_global_id(1); // in col ID

	if (i < M && j < Nin)
	{
		//max pooling size
		int maxpool = Nout / Nin;
		//sum buffer
		float sum = 0;
		//sum over specific network 1 outputs
		for (int k = 0; k < SUM[j + sumsamp]; k++)
		{
			sum += OUT[(j*maxpool+k)*M + i];
		}
		//save the sum in the network 2 input
		IN[M*j + i] = sum;
	}
}

__kernel void Error_distr(const __global float* INerr,
	__global float* OUTerr,
	const __global float* SUM,
	const int M, const int Nin, const int Nout, const int sumsamp)
{
	const int i = get_global_id(0); // in row ID 
	const int j = get_global_id(1); // in col ID

	if (i < M && j < Nout)
	{
		//max pooling size
		int maxpool = Nout / Nin;
		int j_in = j / maxpool;
	
		if (j - maxpool*j_in < SUM[sumsamp + j_in])
		{
			OUTerr[j*M + i] = INerr[M*j_in + i];
		}
		else
		{
			OUTerr[j*M + i] = 0;
		}
		

	}
}


//copy kernel
__kernel void copy_bias_kernel(__global float* A,
	const __global float* B,
	const __global float* data)
{
	int N = data[1];
	int M = data[0];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M + 1 && j < N)
	{
		if (i < M && j < N)
		{
			A[j*(M + 1) + i] = B[j*M + i];
		}
		else
		{
			A[j*(M + 1) + i] = 1;
		}
	}
	
}

__kernel void SQR(__global float* A,
	const __global float* data)
{
	int M = data[0];
	int N = data[1];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		float x = A[j*M + i];
		A[j*M + i] = x*x;
	}
}


//copy transpose kernel
__kernel void copy_transpose_kernel(__global float* A,
	const __global float* B,
	const __global float* data)
{
	int N = data[1];
	int M = data[0];
	const int i = get_global_id(0); // Local row ID 
	const int j = get_global_id(1); // Local col ID
	if (i < M && j < N)
	{
		A[j*M + i] = B[i*N + j];
	}
}

/**Matrix multiplication stuff**/
#define TS 8

__kernel void simple_mult(
	const __global float* A,
	const __global float* Adata,
	const __global float* B,
	const __global float* Bdata,
	__global float* C) 
{
	const int M = Adata[0];
	const int K = Adata[1];
	const int N = Bdata[1];

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
	const int numTiles = floor((float)K / (float)TS)+1;

	for (int t = 0; t<numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;
		
		Asub[col][row] = (tiledCol<K) ? (A[tiledCol*M + globalRow]) : (0);
		Bsub[col][row] = (tiledRow<K) ? (B[globalCol*K + tiledRow]) : (0);

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if(globalRow < M && globalCol < N)
	C[globalCol*M + globalRow] = acc;
}

__kernel void simple_mult_bias(
	const __global float* A,
	const __global float* Adata,
	const __global float* B,
	const __global float* Bdata,
	__global float* C)
{
	const int M = Adata[0];
	const int K = Adata[1];
	const int N = Bdata[1];

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
	const int numTiles = floor((float)K / (float)TS)+1;

	for (int t = 0; t<numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;

		Asub[col][row] = (tiledCol<K) ? (A[tiledCol*M + globalRow]) : (0);
		Bsub[col][row] = (tiledRow<(K - 1)) ? (B[globalCol*(K - 1) + tiledRow]) : ((tiledRow<K) ? 1 : 0);
	

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if (globalRow < M && globalCol < N)
	C[globalCol*M + globalRow] = acc;
}

//special matrix multiplications//
__kernel void simple_mult_bias_activate(
	const __global float* A,
	const __global float* B,
	__global float* C,
	__global float* CA,
	const __global float* A_ID,
	const int M, const int K, const int N)
{
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

	for (int t = 0; t<numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;

		Asub[col][row] = (tiledCol<K) ? (A[tiledCol*M + globalRow]) : (0);
		Bsub[col][row] = (tiledRow<(K - 1)) ? (B[globalCol*(K - 1) + tiledRow]) : ((tiledRow<K) ? 1 : 0);


		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if (globalRow < M && globalCol < N)
	{
		C[globalCol*M + globalRow] = acc;
		CA[globalCol*M + globalRow] = activ(acc, (int)A_ID[globalRow]);
	}
		
}


__kernel void simple_mult_bias_adam(
	const __global float* A,
	const __global float* B,
	__global float* W,
	__global float* Wv,
	__global float* Wm,
	const int Mi, const int K, const int N,
	const float alpha, const float beta1, const float beta2, 
	const int IT
	)
{
	const int M = Mi + 1;
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

	for (int t = 0; t<numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;

		Asub[col][row] = (tiledCol < K) ? ((globalRow != M-1) ? (A[tiledCol*(M - 1) + globalRow]) : (1)) : (0);
		Bsub[col][row] = (tiledRow < K) ? (B[globalCol*K + tiledRow]) : (0);

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if (globalRow < M && globalCol < N)
	{
		int LIT = IT;
		float G = acc/4096;
		if (LIT == 0)
		{
			Wm[globalRow*N + globalCol] = 0;
			Wv[globalRow*N + globalCol] = 0;
		}
		
		float Mg = beta1*Wm[globalRow*N + globalCol] + G*(1 - beta1);
		float Mv = beta2*Wv[globalRow*N + globalCol] + G*G*(1 - beta2); // adam

		Wm[globalRow*N + globalCol] = Mg;
		Wv[globalRow*N + globalCol] = Mv;

		

		if (LIT < 15)
		{
			Mg /= (1 - native_powr(beta1, (float)(LIT + 1)));
			Mv /= (1 - native_powr(beta2, (float)(LIT + 1)));
		}

		//L2 normalization
		float k = 1-0.01*alpha;
		//if bias then dont normalize
		if (globalRow == M - 1)
		{
			k = 1;
		}
		//Weight update
		W[globalRow*N + globalCol] = k*W[globalRow*N + globalCol] - k*alpha * (beta1*Mg + (1 - beta1)*G) / (native_sqrt(beta2*Mv + G*G*(1 - beta2)) + 1e-8);
	}
}


__kernel void simple_mult_error(
	const __global float* A,
	const __global float* B,
	__global float* C,
	const int M, const int K, const int N)
{

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

	for (int t = 0; t<numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;

		Asub[col][row] = (tiledCol<K) ? (A[tiledCol*M + globalRow]) : (0);
		Bsub[col][row] = (tiledRow<K) ? (B[globalCol*K + tiledRow]) : (0);

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if (globalRow < M && globalCol < N-1)
		C[globalRow*(N-1) + globalCol] = acc;
}


__kernel void simple_mult_bias_l(
	const __global float* A,
	const __global float* Adata,
	const __global float* B,
	const __global float* Bdata,
	__global float* C)
{
	const int M = Adata[0]+1;
	const int K = Adata[1];
	const int N = Bdata[1];

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

	for (int t = 0; t<numTiles; t++) {

		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;
	
		Asub[col][row] = (tiledCol < K) ? ((globalRow != M) ? (A[tiledCol*(M - 1) + globalRow]) : (1)) : (0);
		Bsub[col][row] = (tiledRow < K) ? (B[globalCol*K + tiledRow]) : (0);

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result
	if (globalRow < M && globalCol < N)
	C[globalCol*M + globalRow] = acc;
}

//Bitonic sort kernel
__kernel void ParallelBitonic(__global float *data, __global float *buffer, int M, int N, int Np2, int inc, int dir, int copy)
{
	int t = get_global_id(1); // column index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = (t<<1) - low; // insert 0 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order  
	float x0, x1;
	bool swap = 0;

	//loop over 2 columns
	for (int p = 0; p < M; p++)
	{
		if (copy == -1)
		{
			//load from data
			if (inc + i < N)
			{
				x0 = data[(i)*M + p];
				x1 = data[(inc + i)*M + p];
			}
			else
			{
				if (p == 0)
				{
					if (i < N)
					{
						x0 = data[(i)*M + p];
					}
					else
					{
						x0 = HUGE_VALF;
					}
					x1 = HUGE_VALF;
				}
				else
				{
					if (i < N)
					{
						x0 = data[(i)*M + p];
					}
					else
					{
						x0 = 0;
					}
					x1 = 0;
				}
			}
		}
		else
		{
			//use buffer
			x0 = buffer[(i)*M + p];
			x1 = buffer[(inc + i)*M + p];
		}

		if (p == 0)
		{
			swap = reverse ^ (x0 < x1);
		}

		if (swap)
		{
			float b = x1;
			x1 = x0;
			x0 = b;
		}

		// Store 
		if (copy == 1)
		{
			//store to data
			if (i + inc < N)
			{
				data[(i)*M + p] = x0;
				data[(inc + i)*M + p] = x1;
			}
			else if (i < N)
			{
				data[(i)*M + p] = x0;
			}
		}
		else
		{
			//store to buffer
			buffer[(i)*M + p] = x0;
			buffer[(inc + i)*M + p] = x1;
		}
		
	}

}


//Random column mixer
__kernel void Mixer(__global float *data, int M, int N, int lvl, int seed)
{
	int i = get_global_id(0); // row
	int j = get_global_id(1); // column index
	int k0 = j + floor((float)j / (float)lvl)*lvl;
	int k1 = k0 + lvl;
	if (k1 < N)
	{
		float sd = seed+23*j;
		float x0 = data[k0*M + i];
		float x1 = data[k1*M + i];

		if (random(&sd) < 0.5f)
		{
			float b = x1;
			x1 = x0;
			x0 = b;
		}

		data[k0*M + i] = x0;
		data[k1*M + i] = x1;
	}
}
