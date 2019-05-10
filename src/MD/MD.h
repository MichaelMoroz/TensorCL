#pragma once
#include <Tensor.h>
#include <Optimizer.h>
#include <AtomFileLoad.h>
#include <SFML_plot.h>
#include <list>

//molecular dynamics simulation with neural networks
class MD_CL
{
public:
	MD_CL();

	void LoadCluster(Cluster C);
	
	MD_CL(int TypeNum, int N1, int N2);
	MD_CL(std::string filename);

	Tensor GetEnergy(Tensor XYZ, Tensor TYPE);

	float AvgEnergy();

	void PrintEnergies();

	void DecoupleNN(int Iterations);

	void TrainNN(int Iterations, int BatchSize);

	void OptimizeCluster(int cluster_id);
	void ES_Optimization(int cluster_id);
	void MD_Simulation(int cluster_id, float dt);

	void LoadNNFromFile(std::string filename);
	void LoadClusterFromFile(std::string xyzfile, float max_bindenergy = 1000.f);
	void SortTypes();
	void LoadClustersFromFolder(std::string folder, float max_bindenergy = 1000.f);

	void LoadClustersToHostArrays(int random_rot_num);

	void LoadHostToGPU();

private:
	void InitOptimizer();

	//on the GPU
	std::map<int, Tensor> ClustersEnergies;
	std::map<int, Tensor> ClustersXYZ;
	std::map<int, Tensor> ClustersTypes;

	//in RAM
	std::map<int, std::vector< std::vector< std::vector<float> > > > hostClustersXYZ;
	std::map<int, std::vector< std::vector< std::vector<float> > > > hostClustersTypes;
	std::map<int, std::vector<float> > hostClustersEnergies;
	std::map<int, std::vector<float> > hostClustersBindingEnergies;
	std::map<int, std::vector<float> > hostClustersHOMO;
	std::map<int, std::vector<float> > hostClustersLUMO;

	std::vector<int> atom_nums;

	//atom types
	std::map < float , int > types;

	std::vector<Cluster> all_clusters;

	int NN1, NN2, Types;
	float avg_energy;

	//neural network coefficients
	std::vector<Tensor> K;

	//neural network optimizer
	Optimizer OPTIM;
};