#include <MD.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false, 1);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);
		
	//	MD_CL ZnO(2, 16, 16);
	//	ZnO.LoadClustersFromFolder("D:/ZnOTest", 1);
		//ZnO.DecoupleNN(5000);
	//	ZnO.PrintEnergies();
		//ZnO.TrainNN(10000, 2);
		Optimizer OPTIM(Optimizer::GRAD_DESC);
		vector<Tensor> K;
		K.push_back(Tensor(Size(16, 1), 1.f, true));
		K.push_back(Tensor(Size(1, 16), 1.f, true));
		K.push_back(Tensor(Size(16), 0.0f, true));
		K.push_back(Tensor(Size(1), 0.0f, true));

		for (Tensor &W : K)
		{
			OPTIM.AddParameter(W);
			//PrintTensor(W);
		}

	
		
		OPTIM.setSpeed(1e-3);
		int NN = 64;
		Tensor A(Size(1, NN), 3.14f, true);
		Tensor B = sin(A);
	
		for (int i = 0; i < 100; i++)
		{
			Tensor sin_apprx = dot(K[1], dot(K[0], A));
			Tensor COST =  pow(sin_apprx - B, 2);
			OPTIM.Optimize_Cost(COST);
			cout << "COST:" << sum(sum(COST))() << ", Tape size: "<< TAPE_SIZE() << endl;
			for (Tensor &W : K)
			{
				//PrintTensor(W);
			}
		}
		//Tensor sin_apprx = dot(K[1], dot(K[0], A) + repeat(K[2], NN)) + repeat(K[3], NN);
		//PrintTAPE(true);
	}

	system("pause");
	
	return 0;
}