#pragma once
#include <SFML_plot.h>
#include <Tensor.h>
#include <Optimizer.h>

//molecular dynamics simulation with neural networks
class MD_CL
{
public:
	MD_CL();
	MD_CL(int TypeNum, int N1, int N2);
	MD_CL(string filename);

	void LoadNNFromFile(string filename);
	void AddClusterFromFile(string xyzfile);

private:

};