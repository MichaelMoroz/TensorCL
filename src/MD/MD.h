#pragma once
#include <Tensor.h>
#include <Optimizer.h>
#include <AtomFileLoad.h>
#include <SFML_plot.h>

//molecular dynamics simulation with neural networks
class MD_CL
{
public:
	MD_CL();

	void LoadCluster(Cluster C);
	
	MD_CL(int TypeNum, int N1, int N2);
	MD_CL(std::string filename);

	Tensor GetEnergy(int ClusterID);

	void TrainNN(int Iterations, int BatchSize);

	void CalculateClusterEnergy(int cluster_id);
	void OptimizeCluster(int cluster_id);
	void ES_Optimization(int cluster_id);
	void MD_Simulation(int cluster_id, float dt);

	void LoadNNFromFile(std::string filename);
	void LoadClusterFromFile(std::string xyzfile, int rand_rot_num = 0);
	void LoadClustersFromFolder(std::string folder, int rand_rot_num = 100);

private:
	void InitOptimizer();
	int ClusterN;
	std::vector<Tensor> XYZs, Charges;
	std::vector<float> Energies;

	int NN1, NN2, Types;
	std::vector<Tensor> Weights;
	Optimizer OPTIM;
};