#pragma once
#include <iostream>
#include <omp.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <SFML/Graphics.hpp>

extern sf::Font font;

template < typename T > std::string num2str(const T& n)
{
	std::ostringstream stm;
	stm << std::setprecision(7) << floor(100*n)/100.f;
	return stm.str();
}


//Text
bool SetDefaultFont(std::string location);

class Subscript
{
private:
	bool visible;
	sf::Text str;

public:
	Subscript()
	{}


	Subscript(sf::Color color, std::string text);

	void init(sf::Color color, std::string text);

	sf::FloatRect getGlobalBounds();

	void ChangeText(std::string text);

	void ChangeSize(int sz);

	void Draw(sf::RenderWindow *window, float x, float y);

};


float cexp(float x);

float cutlog(float x);

///MATLAB JET COLORMAP
double interpolate(double val, double y0, double x0, double y1, double x1);

double scr(double val);

double red(double gray);

double blue(double gray);

///MATLAB JET COLORMAP

sf::Color float_to_color(float number, float min, float max);

void updateTable(float min, float max);

inline float fastlog10(float x);

///MATLAB JET COLORMAP

float redj(float value, float min, float max);

float greenj(float value, float min, float max);

float bluej(float value, float min, float max);

sf::Color float_to_colorj(float number, float min, float max);

double asy(double He, double k);

float red_complex(float value, float min, float max);

float green_complex(float value, float min, float max);

float blue_complex(float value, float min, float max);

float lin_trunc(float a, float max);

sf::Color float_to_color_complex(float number_real, float number_complex, float min, float max);

void updateTablej(float min, float max);

sf::Color float2color(float number);

sf::Color float2colorj(float number);

void CPU_COLOR_GRAD(sf::Image* img, float** DATA, int dx, int dy);

void CPU_COLOR_GRAD_JET(sf::Image* img, float** DATA, int dx, int dy);

void CPU_COLOR_GRAD_COMPLEX(sf::Image* img, float** DATAR, float** DATAI, int dx, int dy);

void CPU_COLOR_GRAD_2(sf::Image* img, float** DATA1, float** DATA2, int dx, int dy);

void CPU_COLOR_GRAD_3(sf::Image* img, float** DATA1, float** DATA2, int dx, int dy);

class SFMLP
{
private:
	int width, height;
	sf::RenderWindow window;
	std::vector<sf::VertexArray> Lines;
	std::vector<  std::vector<float> > Lines_x, Lines_y;
	std::vector<sf::Color> Lines_color;
	std::vector< std::string> Lines_legend;
	sf::Vector2i mouse_prev;
	int frame, touch;
	sf::Vertex Yaxis[2], Xaxis[2], OneX[2], OneY[2];
	float l_x, l_y, w_x, w_y;
	
	bool imageset;
	int imgX, imgY;
	float imgx0, imgy0, imgx1, imgy1;
	sf::Texture texture;
	sf::Image mainimg;
	sf::Sprite MainSprite;
	sf::Text su;

public:
	bool open;
	SFMLP(int w, int h, float WX, float WY, float lx, float ly);

	~SFMLP()
	{

	};

	void SetImage(float**& A, int W, int H, float x0, float y0, float x1, float y1, float min, float max);

	void AddLine(std::vector<float> X, std::vector<float> Y, sf::Color C, std::string S);

	void AddEmptyLine(sf::Color C, std::string S);

	void AddPointToLine(int l, float x, float y);

	void UpdateState();

	//attention: shitcode ahead

	void DrawAxis(float x0, float x1, float y0, float y1, sf::Color C);

	void DrawGrid(float x0, float x1, float y0, float y1, sf::Color C, sf::Color Te);
};