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

		Tensor A(16, 16), C(16, 16);

		A = random(A);
		C = indicies(C, 0) + 3*indicies(C, 1);
		PrintTensor(A);
		Optimizer root(Optimizer::ADAM);
		root.AddParameter(A);
		root.AddParameter(C);
		
		SFMLP window(1600, 1100, 200, 6, 200 * 0.5 - 1, 0);
		font.loadFromFile("arialbd.ttf");
		
		window.AddEmptyLine(sf::Color::Red, "Tensor B");

		int i = 0;
		while (window.open)
		{
			i++;
			if (i < 100)
			{
				Tensor B = dot(sin(A - C), cos(C + A)) ^ 2;
				root.Optimization_Cost(B);
				root.OptimizationIteration(0.01);

				Tensor COST = sum(sum(B));
				window.AddPointToLine(0, i, COST());
			}
			window.UpdateState();
		}

		PrintTensor(A);
		PrintTensor(C);
		PrintTAPE(false);
	}

	PrintTAPE(false);
	

	system("pause");
	
	return 0;
}