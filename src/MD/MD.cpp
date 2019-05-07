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

	Energies.push_back(C.BindingEnergy);
}

MD_CL::MD_CL(int TypeNum, int N1, int N2) : Types(TypeNum), NN1(N1), NN2(N2)
{
	//first layer
	K.push_back(Tensor(Size(NN1, 3), 1.0f / NN1, true)); //0
	K.push_back(Tensor(Size(NN1, 1), 1.0f / NN1, true)); //1
	K.push_back(Tensor(Size(NN1, 1), 1.0f / NN1, true)); //2
	K.push_back(Tensor(Size(NN1, 1), 1.0f / NN1, true)); //3, bias

	//second layer
	K.push_back(Tensor(Size(NN1, NN1), 1.0f / NN1, true)); //4
	K.push_back(Tensor(Size(NN1, 1), 1.0f / NN1, true)); //5, bias
	
	//third layer
	K.push_back(Tensor(Size(NN2, NN1), 1.0f / NN1, true)); //6
	K.push_back(Tensor(Size(NN2, 1), 1.0f / NN2, true)); //7, bias

	//output energy layer
	K.push_back(Tensor(Size(1, NN2), 1.0f / NN2, true)); //8
	K.push_back(Tensor(Size(1, 1), 1.0f, true)); //9, bias

	InitOptimizer();
}

Tensor MD_CL::GetEnergy(int id)
{
	Tensor xyz = XYZs[id], charge = Charges[id];
	int atom_num = xyz[1];
	Tensor chargepairs = transpose(charge.repeat(charge[0]), 0,2);
	Tensor xyzpairs = xyz.repeat(atom_num); //cooidinates repeated N times, first dimension is X Y Z
	
	Tensor dist = (xyzpairs - transpose(xyzpairs,1,2) + transpose( repeat(diag(Tensor(atom_num, atom_num),1e10f),3) ,0,2) )^(-1.f); //inverted distance between atom pairs
	
	Tensor X = tanh(dot(K[0], dist) + dot(K[1], chargepairs) + dot(K[2], transpose(chargepairs, 1, 2)) + repeat(repeat(K[3], atom_num), atom_num));
	X = sum(tanh(dot(K[4], X) + repeat(repeat(K[5], atom_num), atom_num)));
	X = tanh(dot(K[6], X) + repeat(K[7], atom_num));

	return sum(dot(K[8], X)) + K[9]*xyz[1];
}

void MD_CL::TrainNN(int Iterations, int BatchSize)
{
	float cost = 10;
	for (int it = 0; it < Iterations && !(isnan(cost) || abs(cost)>1e8); it++)
	{
		Tensor COST(Size(1));
		int sh = 0;//+ std::min(rand() % XYZs.size(), XYZs.size() - BatchSize);
		for (int i = sh; i < sh + BatchSize; i++)
		{
			    COST  = ((GetEnergy(i) - Energies[i]))^2;
		}
		OPTIM.Optimize_Cost(COST, false);
		cost = COST();
		//for (auto &k : K)
		//	PrintTensor(k);
		std::cout << "Current cost: " << cost << ", Current tape id: " << COST.ID() << std::endl;
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
	for (auto &k : K)
		OPTIM.AddParameter(k);

	OPTIM.setSpeed(1e-12);
}
