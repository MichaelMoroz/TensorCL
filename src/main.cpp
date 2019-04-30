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

		
		Tensor A(3, 3);

		A = 1;
		PrintTensor(A);
		Optimizer root(Optimizer::GRAD_DESC);
		root.AddParameter(A);

		for (int i = 0; i < 10; i++)
		{
			Tensor B = sin(A)^2;
			//root.Optimization_Cost(B);
			//root.OptimizationIteration(0.5);
			PrintTensor(B);
		}

		PrintTAPE(false);
	}


	PrintTAPE(false);
	/*SFMLP window(1600, 1100, 200, 6, 200 * 0.5 - 1, 0);
	font.loadFromFile("arialbd.ttf");
	int i = 0;
	window.AddEmptyLine(sf::Color::Red, "log10, Average Force, eV/A");
	while (window.open)
	{
		i++;
		window.UpdateState();
	}*/

	system("pause");
	
	return 0;
}