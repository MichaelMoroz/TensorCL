#include <MD.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false, 0);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);
		
		MD_CL ZnO(2, 32, 32);
		ZnO.LoadClustersFromFolder("D:/ZnO", 400.f);
		ZnO.LoadClustersToHostArrays(1);
		ZnO.LoadHostToGPU();
		ZnO.PrintEnergies();
		ZnO.TrainNN(1500, 128);
		ZnO.PrintEnergies();


	/*	Optimizer OPTIM(Optimizer::ADAM);
		vector<Tensor> K;
		K.push_back(Tensor(Size(32, 1), 1.f/32.f, true));
		K.push_back(Tensor(Size(16, 32), 1.f/24.f, true));
		K.push_back(Tensor(Size(1, 16), 1.f/16.f, true));
		K.push_back(Tensor(Size(32), 1.0f, true));
		K.push_back(Tensor(Size(16), 1.0f, true));

		for (Tensor &W : K)
		{
			OPTIM.AddParameter(W);
			//PrintTensor(W);
		}

	
		
		OPTIM.setSpeed(2e-3f);
		int NN = 128;
		Tensor A(Size(1, NN));
		A = indicies(A, 1)*(6.28f/128.f);
		Tensor B = sin(A);
	
		for (int i = 0; i < 1000; i++)
		{
			Tensor sin_apprx = dot(K[2],max(dot(K[1], max(dot(K[0], A) + repeat(K[3], NN))) + repeat(K[4], NN)));
			Tensor dif = sin_apprx - B;
			Tensor COST = pow(dif, 2);
		    OPTIM.Optimize_Cost(COST);
			cout << "COST:" << sum(sum(COST))() << ", Tape size: "<< TAPE_SIZE() << endl;
			
			if (i == 999)
			{
				PrintTensor(dif);
			}
		}*/
	}	

	system("pause");
	
	return 0;
}