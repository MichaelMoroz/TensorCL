#include <SFML_plot.h>
#include <Tensor.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", false);

	if (!cl.failed)
	{
		TensorUseOpenCL(&cl);

		Tensor A(3, 3), B(3, 3);

		A = (indicies(A, 0) + 1 + indicies(A, 1))/10.f;
		PrintTensor(A);
		B = indicies(B, 1) + indicies(B, 0)+1.f;
		Tensor dB = _if(B<2, B, 0.f);
		B = B + dB;
		PrintTensor(B);
		PrintTensor(dB);

		Tensor SIN = sin(A) + cos(B)*0.5f;

		PrintTensor(SIN);

		Tensor DOT = dot(SIN, B);
		PrintTensor(DOT);

		Gradient dDOT(DOT);

		PrintTensor(dDOT.wrt(A));
		PrintTensor(dDOT.wrt(B));

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