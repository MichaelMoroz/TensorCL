#include "MD.h"
#include <algorithm>
MD_CL::MD_CL()
{
}

MD_CL::MD_CL(int TypeNum, int N1, int N2) : Types(TypeNum), NN1(N1), NN2(N2)
{
	//first layer
	Tensor C(Size(1));//uh

	K.push_back(Tensor(Size(NN1, 3), 1.f, true)); //0
	K.push_back(Tensor(Size(NN1, TypeNum), 0.01f / NN1, true)); //1
	K.push_back(Tensor(Size(NN1, TypeNum), 0.01f / NN1, true)); //2
	K.push_back(Tensor(Size(NN1, 1), 1.f/ NN1, true)); //3, bias
	
	//third layer
	K.push_back(Tensor(Size(NN2, NN1), 0.2f/(NN1+NN2), true)); //4
	K.push_back(Tensor(Size(NN2, 1), 1.0f / (NN1 + NN2), true)); //5, bias

	//output energy layer
	K.push_back(Tensor(Size(1, NN2), 100.f / NN2, true)); //6

	InitOptimizer();
}

MD_CL::MD_CL(std::string filename)
{
}

Tensor sinx(Tensor& X)
{
	return X + sin(X);
}

Tensor MD_CL::GetEnergy(Tensor XYZ, Tensor TYPE)
{
	int atom_num = XYZ[1]; int cluster_num = XYZ[2];

	Tensor chargepairs = transpose(TYPE.repeat(atom_num), 2, 3);

	//cooidinates repeated N times, first dimension is X Y Z
	Tensor xyzpairs = transpose(XYZ.repeat(atom_num), 2, 3); 
	
	Tensor diagonal = repeat( transpose(repeat(diag(Tensor(atom_num, atom_num), 1e10f), 3), 0, 2) , cluster_num);
	//inverted distance between atom pairs
	Tensor dist = pow(xyzpairs - transpose(xyzpairs,1,2) + diagonal, -2.f); 
//	PrintTAPE(false);
//	std::cout << K[0].ID() << std::endl;
	Tensor X = tanh(dot(K[0], dist) + repeat(multirepeat(K[3], atom_num, 2), cluster_num) + dot(K[1], chargepairs) + dot(K[2], transpose(chargepairs, 1, 2)));
	X = tanh(dot(K[4], sum(transpose(X, 2, 3))) + repeat(repeat(K[5], atom_num), cluster_num));

	return sum(transpose(dot(K[6], X),1,2)) + atom_num*avg_energy;
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

void MD_CL::PrintEnergies()
{
	for (int i = 0; i < atom_nums.size(); i++)
	{
		Tensor E = GetEnergy(ClustersXYZ[atom_nums[i]], ClustersTypes[atom_nums[i]]);
		PrintTensor( E - transpose(ClustersEnergies[atom_nums[i]]) );
	}
}

void MD_CL::TrainNN(int Iterations, int BatchSize, float min_cost)
{
	//create the training batches
	std::vector<Tensor> BATCHES_XYZ, BATCHES_TYPES, BATCHES_ENERGIES;

	for (int i = 0; i < atom_nums.size(); i++)
	{
		int clusters = ClustersXYZ[atom_nums[i]][2];
		int cur_batchnum = clusters / BatchSize;
		for (int j = 0; j < cur_batchnum-1; j++)
		{
			BATCHES_XYZ.push_back(ClustersXYZ[atom_nums[i]].cut(j*BatchSize, (j + 1)*BatchSize));
			BATCHES_TYPES.push_back(ClustersTypes[atom_nums[i]].cut(j*BatchSize, (j + 1)*BatchSize));
			BATCHES_ENERGIES.push_back(transpose(ClustersEnergies[atom_nums[i]].cut(j*BatchSize, (j + 1)*BatchSize)));
		}
	}

	float avgcost = 110.f*110.f;
	int epoch = 0;
	for (int it = 0; it < Iterations && isfinite(avgcost) && avgcost > min_cost; it++)
	{
		int batch = rand() % BATCHES_XYZ.size();

		Tensor E = GetEnergy(BATCHES_XYZ[batch], BATCHES_TYPES[batch]);
	    Tensor COST = pow(E - BATCHES_ENERGIES[batch],2.f)/ BatchSize;

		OPTIM.Optimize_Cost(COST);
		float cur_cost = sum(COST)();
		if (it%BATCHES_XYZ.size()==0) epoch++;
		std::cout <<"Epoch: "<< epoch << ", Current cost: " << sqrt((avgcost = (it==0)? cur_cost :(avgcost * 0.95 + cur_cost * 0.05))) << ", Current tape size: " << TAPE_SIZE() << std::endl;
	}
}

void MD_CL::LoadNNFromFile(std::string filename)
{
	OPTIM.Clear();
	K.clear();
	std::ifstream file(filename, std::ios::binary);

	while (!file.eof())
	{
		K.push_back(Tensor(TensorCL(file)));
	}
	
	file.close();
	InitOptimizer();
}

void MD_CL::SaveNNToFile(std::string filename)
{
	std::ofstream file(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	for (auto &k : K)
		k.GetTensor().SaveToFstream(file);
	file.close();
}

void MD_CL::PrintCoefficients()
{
	for (auto &k : K)
		PrintTensor(k);
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
	SortTypes();
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
	avg_energy = 0;
	for (auto it = hostClustersXYZ.begin(); it != hostClustersXYZ.end(); it++)
	{
		TensorData xyz(3, it->first, it->second.size());
		TensorData id(2, it->first, it->second.size());
		TensorData E(it->second.size());
		xyz.LoadData(it->second);
		id.LoadData(hostClustersTypes[it->first]);
		E.LoadData(hostClustersEnergies[it->first]);

		atom_nums.push_back(it->first);

		ClustersXYZ[it->first] = Tensor(xyz);
		ClustersTypes[it->first] = Tensor(id);
		ClustersEnergies[it->first] = Tensor(E);

		avg_energy += vectoravg(hostClustersEnergies[it->first])/(float) it->first;
	}
	avg_energy /= (float)hostClustersXYZ.size();
}

void MD_CL::InitOptimizer()
{
	OPTIM.setMethod(Optimizer::ADAM);
	//OPTIM.setRegularization(Optimizer::L2, 0.005);
	for (auto &k : K)
		OPTIM.AddParameter(k);

	OPTIM.setSpeed(0.002);
}
