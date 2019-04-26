#include <SFML_plot.h>
#include <Tensor.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);

		Tensor A(3, 5), B(5, 2, 2);

		A = indicies(A, 0) + 1;
		PrintTensor(A);
		B = indicies(B, 1) + 1 + indicies(B, 2);
		PrintTensor(B);

		Tensor DOT = dot(A, B);

		PrintTensor(DOT);
		PrintTensor(transpose(DOT, 0, 2));
		PrintTAPE(false);
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