#include <SFML_plot.h>
#include <CL_TENSOR.h>

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	OpenCL cl("OpenCL/main.c", 0, false);

	TensorUseOpenCL(&cl);

	TensorCL A(3, 3), B(3, 3);

	TensorCL C = A + B;

	SFMLP window(1600, 1100, 200, 6, 200 * 0.5 - 1, 0);
	font.loadFromFile("arialbd.ttf");
	int i = 0;
	window.AddEmptyLine(sf::Color::Red, "log10, Average Force, eV/A");
	while (window.open)
	{
		i++;
		window.UpdateState();
	}
	
	return 0;
}