#include <MD.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);
		
		MD_CL ZnO(2, 16, 16);
		ZnO.LoadClustersFromFolder("E:/ZnOTest", 1);
		ZnO.DecoupleNN(5000);
		ZnO.PrintEnergies();
		ZnO.TrainNN(10000, 2);
		/*Optimizer OPTIM(Optimizer::ADAM);
		vector<Tensor> K;
		K.push_back(Tensor(Size(2, 4), 1.f, true));
		K.push_back(Tensor(Size(3, 2), 1.f, true));
		K.push_back(Tensor(Size(2), 0.f, true));
		K.push_back(Tensor(Size(3), 0.f, true));

		for (Tensor &W : K)
		{
			OPTIM.AddParameter(W);
			//PrintTensor(W);
		}

	
		
		OPTIM.setSpeed(0.06);

		Tensor A(Size(4, 4, 2), 0.2f, true), B(Size(3,4),0.2f,true);
		Tensor C(Size(4, 4, 2), 0.2f, true), D(Size(3, 4), 0.2f, true);
	
		for (int i = 0; i < 10000; i++)
		{
			Tensor COST(Size(3, 4));
			COST = COST + (dot(K[1], sum(max(dot(K[0], A) + repeat(repeat(K[2], 4), 2)))) + repeat(K[3], 4) - B) ^ 2;
			COST = COST + (dot(K[1], sum(max(dot(K[0], C) + repeat(repeat(K[2], 4), 2)))) + repeat(K[3], 4) - D) ^ 2;
			OPTIM.Optimize_Cost(COST);
			cout << "COST:" << sum(sum(COST))() << ", Tape size: "<< TAPE_SIZE() << endl;
			for (Tensor &W : K)
			{
				//PrintTensor(W);
			}
		}

		PrintTAPE(false);*/
	}

	system("pause");
	
	return 0;
}