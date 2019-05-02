#include <SFML_plot.h>
#include <Tensor.h>
#include <Optimizer.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);
		int N = 128;
		Tensor A(N, N), C(N, N);

		A = random(A);
		C = indicies(C, 0) + 3*indicies(C, 1);
		//PrintTensor(A);
		Optimizer root(Optimizer::ADAM);
		root.AddParameter(A);
		root.AddParameter(C);
		
		SFMLP window(1600, 1100, 200, 6, 200 * 0.5 - 1, 0);
		
		window.AddEmptyLine(sf::Color::Red, "Tensor B");
		window.AddEmptyLine(sf::Color::Blue, "TensorTapeSize");

		int i = 0;
		while (window.open)
		{
			i++;
			if (i < 10000)
			{
				Tensor B = dot(sin(A - C), cos(C + A)) ^ 2;
				root.Optimization_Cost(B);
				root.OptimizationIteration(1.f/(N*N));

				Tensor COST = sum(sum(B));
				window.AddPointToLine(0, i, COST()/(N * N));
			}
			window.AddPointToLine(1, i, TAPE_SIZE());
			window.UpdateState();
		}

		//PrintTensor(A);
		//PrintTensor(C);
		PrintTAPE(false);
	}

	PrintTAPE(false);
	

	system("pause");
	
	return 0;
}