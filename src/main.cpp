#include <SFML_plot.h>
#include <Tensor.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);

		Tensor A(2, 3);

		A = (indicies(A, 0) + 1 + indicies(A, 1))/10.f;
		PrintTensor(A);

		Tensor SIN = sin(A) + A*2.f;

		PrintTensor(SIN);

		Gradient dSIN(SIN);

		PrintTensor(dSIN.wrt(A));

		PrintTAPE(true);
	}
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