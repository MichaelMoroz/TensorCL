#include <MD.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);
		
		//MD_CL ZnO(2, 8, 8);
		//ZnO.LoadClusterFromFile("E:/0.xyz");
		//ZnO.TrainNN(100, 1);
		vector<Tensor> K;
		K.push_back(Tensor(Size(2, 4), 1.f, true));
		K.push_back(Tensor(Size(3, 2), 1.f, true));

		Optimizer OPTIM(Optimizer::ADAM);
		Tensor A(Size(4, 4, 2), 1.f), B(Size(3,4),1.f,true);
		A = indicies(A, 0) + 4 * indicies(A, 1) + 10 * indicies(A, 2);
		for (Tensor &W : K)
		{
			OPTIM.AddParameter(W);
		}

		for (int i = 0; i < 1000; i++)
		{
			Tensor COST = (dot(K[1], sum(tanh(dot(K[0], A)))) - B) ^ 2;
			//PrintTAPE(true);
			OPTIM.Optimization_Cost(COST);
			OPTIM.OptimizationIteration(0.001);
			cout << "COST:" << sum(sum(COST))() << endl;
			for (Tensor &W : K)
			{
				PrintTensor(W);
			}
		}
	}

	system("pause");
	
	return 0;
}