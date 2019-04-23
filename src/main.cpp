#include <SFML_plot.h>
#include <CL_TENSOR.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c");

	TensorUseOpenCL(&cl);

	TensorCL A(3, 5), B(5, 2, 2, 2);

	A = 1;
	B = indicies(B);

	PrintTensor(A);
	PrintTensor(B);

	TensorCL DOT = dot(A, B);

	PrintTensor(DOT);
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