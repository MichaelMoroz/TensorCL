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
		Optimizer OPTIM(Optimizer::ADAM);
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

	
		
		OPTIM.setSpeed(1e-3f);
		int NN = 128;
		Tensor A(Size(1, NN));
		A = indicies(A, 1)*(6.28f/128.f);
		Tensor B = sin(A);
	
		for (int i = 0; i < 1000; i++)
		{
			Tensor sin_apprx = dot(K[1], tanh(dot(K[0], A) + repeat(K[2], NN))) + repeat(K[3], NN);
			Tensor COST =  pow(sin_apprx - B, 2);
		    OPTIM.Optimize_Cost(COST);
			cout << "COST:" << sum(sum(COST))() << ", Tape size: "<< TAPE_SIZE() << endl;
			for (Tensor &W : K)
			{
				//PrintTensor(W);
			}
		}
		
		Tensor sin_apprx = dot(K[1], tanh(dot(K[0], A) + repeat(K[2], NN))) + repeat(K[3], NN);
		PrintTensor(sin_apprx - B);
	}	

	system("pause");
	
	return 0;
}