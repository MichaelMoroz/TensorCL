#include "MD.h"
#include <algorithm>
MD_CL::MD_CL()
{
}

void MD_CL::LoadCluster(Cluster C)
{
	TensorData xyz(3, C.atom_coords.size());
	TensorData id(C.atom_coords.size(), 1);

	xyz.LoadData(C.atom_coords);
	id.LoadData(C.atom_id);

	XYZs.push_back(Tensor(xyz));
	Charges.push_back(Tensor(id));

	Energies.push_back(C.Energy);
}

MD_CL::MD_CL(int TypeNum, int N1, int N2) : Types(TypeNum), NN1(N1), NN2(N2)
{
	
	//BIASES
	Tensor B1L1(NN1, 1);
	Tensor B1L2(NN1, 1);
	Tensor B2L1(NN2, 1);
	Tensor B2L2(NN2, 1);
	Tensor B2L3(1, 1);

	//Input layer
	Tensor dXYZL1(NN1, 3); //inverted distances to first layer
	Tensor T1L1(NN1, 1); //atom 1 type
	Tensor T2L1(NN1, 1); //atom 2 type
	dXYZL1 = random(dXYZL1);
	T1L1 = random(T1L1);
	T2L1 = random(T2L1);
	Weights.push_back(dXYZL1);
	Weights.push_back(T1L1);
	Weights.push_back(T2L1);
	Weights.push_back(B1L1);
	

	//Rest of the layers
	Tensor N1L2(NN1, NN1); N1L2 = random(N1L2);
	Tensor N2L1(NN1, NN2); N2L1 = random(N2L1);
	Tensor N2L2(NN2, NN2); N2L2 = random(N2L2);
	Tensor N2L3(1, NN2); N2L3 = random(N2L3);

	Weights.push_back(N1L2);
	Weights.push_back(B1L2);

	Weights.push_back(N2L1);
	Weights.push_back(B2L1);

	Weights.push_back(N2L2);
	Weights.push_back(B2L2);

	Weights.push_back(N2L3);
	Weights.push_back(B2L3);

	InitOptimizer();
}

Tensor MD_CL::GetEnergy(int id)
{
	Tensor xyz = XYZs[id], charge = Charges[id];
	Tensor chargepairs = transpose(charge.repeat(charge[0]), 0,2);
	Tensor xyzpairs = xyz.repeat(xyz[1]); //cooidinates repeated N times, first dimension is X Y Z
	Tensor dist = transpose((xyzpairs - transpose(xyzpairs,1,2))^(-1)); //inverted distance between atom pairs
	Tensor NETWORK1LAYER1 = max(dot(Weights[0], dist) + dot(Weights[1], chargepairs) + 
		dot(Weights[2], transpose(chargepairs, 1, 2)) + repeat(repeat(Weights[3], xyz[1]), xyz[1]));
	//sum over all atom pairs
	Tensor NETWORK1LAYER2 = sum(max(dot(Weights[4], NETWORK1LAYER1) + repeat(repeat(Weights[5], xyz[1]), xyz[1])));
	Tensor NETWORK2LAYER1 = max(dot(Weights[5], NETWORK1LAYER2) + repeat(Weights[6], xyz[1]));
	Tensor NETWORK2LAYER2 = max(dot(Weights[7], NETWORK2LAYER1) + repeat(Weights[8], xyz[1]));
	//return energy + energy shift
	return sum(dot(Weights[9], NETWORK2LAYER2)) + Weights[10]*xyz[1];
}

void MD_CL::TrainNN(int Iterations, int BatchSize)
{
	for (int it = 0; it < Iterations; it++)
	{
		Tensor COST((unsigned int)1, (unsigned int)1);
		int sh = std::min(rand() % XYZs.size(), XYZs.size() - BatchSize);
		for (int i = sh; i < sh + BatchSize; i++)
		{
			COST = COST + (GetEnergy(i) - Energies[i]) ^ 2;
		}
		OPTIM.Optimization_Cost(COST);
		OPTIM.OptimizationIteration(0.05);
		std::cout << "Current cost: " << COST() << std::endl;
	}
}

void MD_CL::LoadClusterFromFile(std::string xyzfile, int rand_rot_num)
{
	Cluster C = XYZ_Load(xyzfile);
	if (rand_rot_num > 0)
	{
		for (int i = 0; i < rand_rot_num; i++)
		{
			LoadCluster(RandomRotateCluster(C));
		}
	}
	else
	{
		LoadCluster(C);
	}
}

void MD_CL::LoadClustersFromFolder(std::string folder, int rand_rot_num)
{
	std::vector < fs::path > paths = GetFilesInFolder(folder, ".xyz");

	for (auto &path : paths)
	{
		std::string file = path.generic_string();
		LoadClusterFromFile(file, rand_rot_num);
	}
}

void MD_CL::InitOptimizer()
{
	OPTIM.setMethod(Optimizer::GRAD_DESC);
	for (auto &k : Weights)
		OPTIM.AddParameter(k);
}
