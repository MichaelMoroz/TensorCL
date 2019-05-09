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
	K.push_back(Tensor(Size(NN1, 3), 0.33f, true)); //0
	K.push_back(Tensor(Size(NN1, 1), 0.01f / NN1, true)); //1
	K.push_back(Tensor(Size(NN1, 1), 0.01f / NN1, true)); //2
	K.push_back(Tensor(Size(NN1, 1), 1.f/ NN1, true)); //3, bias

	//second layer
	K.push_back(Tensor(Size(NN1, NN1), 0.3f / NN1, true)); //4
	K.push_back(Tensor(Size(NN1, 1), 1.f / NN1, true)); //5, bias
	
	//third layer
	K.push_back(Tensor(Size(NN2, NN1), 0.2f/(NN1+NN2), true)); //6
	K.push_back(Tensor(Size(NN2, 1), 1.0f / (NN1 + NN2), true)); //7, bias

	//output energy layer
	K.push_back(Tensor(Size(1, NN2), 0.3f / NN2, true)); //8

	InitOptimizer();
}

Tensor sinx(Tensor& X)
{
	return X + sin(X);
}

Tensor MD_CL::GetEnergy(int id)
{
	Tensor xyz = XYZs[id], charge = Charges[id];
	int atom_num = xyz[1];
	Tensor chargepairs = transpose(charge.repeat(charge[0]), 0,2);
	Tensor xyzpairs = xyz.repeat(atom_num); //cooidinates repeated N times, first dimension is X Y Z
	
	Tensor dist = pow(xyzpairs - transpose(xyzpairs,1,2) + transpose( repeat(diag(Tensor(atom_num, atom_num),1e10f),3) ,0,2), -1.f); //inverted distance between atom pairs
	
	Tensor X = sin(dot(K[0], dist) + repeat(repeat(K[3], atom_num), atom_num) + dot(K[1], chargepairs) + dot(K[2], transpose(chargepairs, 1, 2)));
	X = sum(sin(dot(K[4], X) + repeat(repeat(K[5], atom_num), atom_num)));
	X = sin(dot(K[6], X) + repeat(K[7], atom_num));

	return sum(dot(K[8], X)) + avg_energy;
}

float vectoravg(std::vector<float> &a)
{
	float avg = 0.f;
	for (auto &x : a)
	{
		avg += x;
	}
	return avg / (float)a.size();
}

float MD_CL::AvgEnergy()
{
	float avg = 0.f;
	for (int i = 0; i < Energies.size(); i++)
	{
		Tensor E = GetEnergy(i);
		avg += E();
	}
	return avg / (float)Energies.size();
}

void MD_CL::PrintEnergies()
{
	for (int i = 0; i < Energies.size(); i++)
	{
		Tensor E = GetEnergy(i);
		std::cout << "Energy: " << i << " " << E() << " " << Energies[i] << std::endl;
	}
}

void MD_CL::DecoupleNN(int Iterations)
{
	float cost = 0.f, costsmooth = 0.f;
	float avgenergy = AvgEnergy();
	OPTIM.setSpeed(0.003f);
	for (int it = 0; it < Iterations && isfinite(cost);)
	{
		int i = rand() % XYZs.size();
		int j = rand() % XYZs.size();
		if (i != j)
		{
			Tensor E1 = GetEnergy(i);
			Tensor E2 = GetEnergy(j);
			Tensor COST = -pow((E1 - E2) / avgenergy, 2.f);
			OPTIM.Optimize_Cost(COST);
			avgenergy = avgenergy * 0.9f + (E1() + E2())*0.1f / 2.f;
			cost = COST();
			if (it == 0) costsmooth = cost;
			costsmooth = costsmooth * 0.95 + cost * 0.05;
			std::cout << "Current cost: " << costsmooth << std::endl;
			it++;
		}
	}
	OPTIM.setSpeed(1e-5f);
}

void MD_CL::TrainNN(int Iterations, int BatchSize)
{
	float avgcost = 0;
	avg_energy = vectoravg(Energies);

	for (int it = 0; it < Iterations && isfinite(avgcost); it++)
	{
		Tensor COST(Size(1));
		int sh = std::min(rand() % XYZs.size(), XYZs.size() - BatchSize);
		for (int i = sh; i < std::min(sh + BatchSize,(int)Energies.size()); i++)
		{
			Tensor E = GetEnergy(i);
		    COST  = COST + pow(E - Energies[i],2.f);
		}
		COST /= BatchSize;
		OPTIM.Optimize_Cost(COST);
		std::cout << "Current cost: " << (avgcost = (it==0)?COST():(avgcost * 0.95 + COST() * 0.05)) << ", Current tape id: " << COST.ID() << std::endl;
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
	OPTIM.setMethod(Optimizer::ADAM);
	//OPTIM.setRegularization(Optimizer::L2, 0.003);
	for (auto &k : K)
		OPTIM.AddParameter(k);

	OPTIM.setSpeed(0.001);
}
