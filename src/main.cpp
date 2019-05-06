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
		OPTIM.setSpeed(0.5);

		Tensor A(Size(4, 4, 2), 0.2f, true), B(Size(3,4),0.2f,true);
		for (Tensor &W : K)
		{
			OPTIM.AddParameter(W);
			//PrintTensor(W);
		}

		for (int i = 0; i < 10000; i++)
		{
			Tensor COST = (dot(K[1], sum(max(dot(K[0], A)))) - B) ^ 2;
			OPTIM.Optimize_Cost(COST);
			cout << "COST:" << sum(sum(COST))() << ", Tape size: "<< TAPE_SIZE() << endl;
			for (Tensor &W : K)
			{
				//PrintTensor(W);
			}
		}

		PrintTAPE(false);
	}

	system("pause");
	
	return 0;
}