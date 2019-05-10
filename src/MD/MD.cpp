#include "MD.h"
#include <algorithm>
MD_CL::MD_CL()
{
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

MD_CL::MD_CL(std::string filename)
{
}

Tensor sinx(Tensor& X)
{
	return X + sin(X);
}
/*
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
*/
bool list_has_element(std::list<int>& L, int E)
{
	return std::count(L.begin(), L.end(), E)>0;
}

int find_element(std::list<int>& L, int E)
{
	return std::distance(L.begin(), std::find(L.begin(), L.end(), E));
}

void MD_CL::LoadClusterFromFile(std::string xyzfile, float max_bindenergy)
{
	Cluster C = XYZ_Load(xyzfile);
	if (C.BindingEnergy < max_bindenergy)
	{
		all_clusters.push_back(C);

		//add atom types
		for (float &i : C.atom_id)
		{	
			types[i] = 1;
		}
	}
}

void MD_CL::SortTypes()
{
	int i = 0;
	for (auto it = types.begin(); it != types.end(); it++, i++)
	{
		it->second = i;
	}
}

void MD_CL::LoadClustersFromFolder(std::string folder, float max_bindenergy)
{
	for (auto &path : GetFilesInFolder(folder, ".xyz"))
	{
		std::string file = path.generic_string();
		LoadClusterFromFile(file, max_bindenergy);
	}
}

void MD_CL::LoadClustersToHostArrays(int random_rot_num)
{
	for (int i = 0; i < all_clusters.size()*random_rot_num; i++)
	{
		int id = (rand() % all_clusters.size());
		Cluster rotated = RandomRotateCluster(all_clusters[id]);

		int atoms = rotated.atom_id.size();
		std::vector< std::vector<float> > atom_types;
		for (int a = 0; a < atoms; a++)
		{
			std::vector<float> type;
			for (int b = 0; b < types.size(); b++)
			{
				int yes = b == types[rotated.atom_id[a]];
				type.push_back(yes);
			}
			atom_types.push_back(type);
		}

		hostClustersEnergies[atoms].push_back(rotated.Energy);
		hostClustersBindingEnergies[atoms].push_back(rotated.BindingEnergy);
		hostClustersHOMO[atoms].push_back(rotated.HOMO);
		hostClustersLUMO[atoms].push_back(rotated.LUMO);
		hostClustersTypes[atoms].push_back(atom_types);
		hostClustersXYZ[atoms].push_back(rotated.atom_coords);
	}
}

void MD_CL::LoadHostToGPU()
{
	for (auto it = hostClustersXYZ.begin(), it2 = hostClustersTypes.begin(); it != hostClustersXYZ.end(); it++, it2++)
	{
		TensorData xyz(3, it->first, it->second.size());
		TensorData id(2, it->first, it->second.size());
		TensorData E(it->second.size());
		xyz.LoadData(it->second);
		id.LoadData(it2->second);
		E.LoadData(hostClustersEnergies[it->first]);

		atom_nums.push_back(it->first);

		ClustersXYZ[it->first] = Tensor(xyz);
		ClustersTypes[it->first] = Tensor(id);
		ClustersEnergies[it->first] = Tensor(E);
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
