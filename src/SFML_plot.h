#pragma once
#include <iostream>
#include <omp.h>
#include <iomanip>
using namespace sf;


template < typename T > std::string num2str(const T& n)
{
	std::ostringstream stm;
	stm << std::setprecision(7) << floor(100*n)/100.f;
	return stm.str();
}


Font font;
//Text
bool SetDefaultFont(string location)
{
	if (font.loadFromFile(location))
	{
		return true;
	}
	else return false;
}

class Subscript
{
private:
	bool visible;
	Text str;

public:
	Subscript()
	{}


	Subscript(Color color, string text)
	{
		str.setFont(font);
		str.setCharacterSize(24);
		str.setColor(color);
		str.setString(text);
		visible = 1;
	}

	void init(Color color, string text)
	{
		str.setFont(font);
		str.setCharacterSize(24);
		str.setColor(color);
		str.setString(text);
		visible = 1;
	}

	FloatRect getGlobalBounds()
	{
		return str.getGlobalBounds();
	}

	void ChangeText(string text)
	{
		str.setString(text);
	}

	void ChangeSize(int sz)
	{
		str.setCharacterSize(sz);
	}

	void Draw(RenderWindow *window, float x, float y)
	{
		str.setPosition(x, y);
		if (visible)
			window->draw(str);
	}

};

float xm = 1.5;
float cexp(float x)
{
	return (x < -xm) ? (exp(-xm)*(1 + x + xm)) : ((x > xm) ? (exp(xm)*(1 + x - xm)) : (exp(x)));
}

float cutlog(float x)
{
	return (x < exp(-xm)) ? (-xm + exp(xm)*x - 1) : ((x > exp(xm)) ? (xm + exp(-xm)*x - 1) : (log(x)));
}

///MATLAB JET COLORMAP
double interpolate(double val, double y0, double x0, double y1, double x1)
{
	return (val - x0)*(y1 - y0) / (x1 - x0) + y0;
}

double scr(double val)
{
	if (val>0) return val;
	else return 0;
}


double red(double gray)
{
	return interpolate(gray, 1.0, -1, 0, 0);
}

double blue(double gray)
{
	return interpolate(gray, 0, 0, 1, 1);
}
///MATLAB JET COLORMAP

Color float_to_color(float number, float min, float max)
{

	if (number<max && number >= min)
	{
		float R, G, B;
		float grayscale = 2 * (number - min) / (max - min) - 1;
		R = scr(red(grayscale));
		G = 0;
		B = scr(blue(grayscale));
		return Color(255 * R, 255 * G, 255 * B);
	}
	if (number >= max) return Color(127, 0, 0);
	if (number < min) return Color(0, 0, 127);
}

int colortablesize = 4000;

Color* ColorTable = new Color[colortablesize];

float def_min, def_max;

void updateTable(float min, float max)
{
#pragma omp parallel for
	for (int i = 0; i < colortablesize; i++)
	{
		ColorTable[i] = float_to_color(min + (float)(max - min)*(float)i / (float)colortablesize, min, max);
	}
	def_min = min;
	def_max = max;
}

float log102 = log10f(2);

inline float fastlog10(float x)
{
	union
	{
		float f;
		int i;
	} vx = { x };
	union
	{
		int i;
		float f;
	} mx = { (vx.i & 0x007FFFFF) | (0x7e << 23) };
	float y = vx.i;
	y *= 1.0 / (1 << 23);

	return log102*(y - 124.22544637f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f));
}


///MATLAB JET COLORMAP

float redj(float value, float min, float max)
{
	float A = 6 * (value - min) / (max - min);
	if (A >= 0 && A<2)
	{
		return 0;
	}
	if (A >= 2 && A<3)
	{
		return interpolate(A, 0, 2, 255, 3);
	}
	if (A >= 3 && A<6)
	{
		return 255;
	}
	if (A <= 0)
		return 0;
	if (A >= 6)
		return 255;
}

float greenj(float value, float min, float max)
{
	float A = 6 * (value - min) / (max - min);
	if (A >= 0 && A<1)
	{
		return 0;
	}
	if (A >= 1 && A<2)
	{
		return interpolate(A, 0, 1, 255, 2);
	}
	if (A >= 2 && A<3)
	{
		return 255;
	}
	if (A >= 3 && A<4)
	{
		return interpolate(A, 255, 3, 0, 4);
	}
	if (A >= 4 && A<5)
	{
		return 0;
	}
	if (A >= 5 && A<6)
	{
		return interpolate(A, 0, 5, 255, 6);
	}
	if (A <= 0)
		return 0;
	if (A >= 6)
		return 255;
}

float bluej(float value, float min, float max)
{
	float A = 6 * (value - min) / (max - min);
	if (A>0 && A<1)
	{
		return interpolate(A, 0, 0, 255, 1);
	}
	if (A >= 1 && A<2)
	{
		return 255;
	}
	if (A >= 2 && A<3)
	{
		return interpolate(A, 255, 2, 0, 3);
	}
	if (A >= 3 && A<5)
	{
		return 0;
	}
	if (A >= 5 && A<6)
	{
		return interpolate(A, 0, 5, 255, 6);
	}
	if (A <= 0)
		return 0;
	if (A >= 6)
		return 255;
}



Color float_to_colorj(float number, float min, float max)
{
	if (number<max && number >= min)
	{
		float R, G, B;
		float grayscale = 2 * (number - min) / (max - min) - 1;
		R = redj(grayscale, -1, 1);
		G = greenj(grayscale, -1, 1);
		B = bluej(grayscale, -1, 1);
		return Color(R, G, B);
	}
	if (number >= max) return Color(0, 0, 0);
	if (number < min) return Color(255, 255, 255);
}

double asy(double He, double k)
{
	He *= k;
	return ((He - 1) / (2 * (sqrt(He*He) + 1)) + 0.5);
}


float red_complex(float value, float min, float max)
{
	float A = 5 * (value - min) / (max - min);
	if (A >= 0 && A<1)
	{
		return interpolate(A, 255, 0, 0, 1);
	}
	if (A >= 1 && A<2)
	{
		return 0;
	}
	if (A >= 2 && A<3)
	{
		return interpolate(A, 0, 2, 255, 3);
	}
	if (A >= 3 && A <= 4)
	{
		return 255;
	}
	if (A >= 4 && A <= 5)
	{
		return interpolate(A, 255, 4, 255, 5);
	}
	if (A <= 0)
		return 255;
	if (A >= 5)
		return 255;
}

float green_complex(float value, float min, float max)
{
	float A = 5 * (value - min) / (max - min);
	if (A >= 0 && A<1)
	{
		return interpolate(A, 0, 0, 0, 1);
	}
	if (A >= 1 && A<2)
	{
		return interpolate(A, 0, 1, 255, 2);
	}
	if (A >= 2 && A<3)
	{
		return 255;
	}
	if (A >= 3 && A<4)
	{
		return interpolate(A, 255, 3, 0, 4);
	}
	if (A >= 4 && A <= 5)
	{
		return interpolate(A, 0, 4, 0, 5);
	}
	if (A <= 0)
		return 0;
	if (A >= 5)
		return 0;
}

float blue_complex(float value, float min, float max)
{
	float A = 5 * (value - min) / (max - min);
	if (A>0 && A<1)
	{
		return interpolate(A, 255, 0, 255, 1);
	}
	if (A >= 1 && A<2)
	{
		return 255;
	}
	if (A >= 2 && A<3)
	{
		return interpolate(A, 255, 2, 0, 3);
	}
	if (A >= 3 && A<4)
	{
		return 0;
	}
	if (A >= 4 && A <= 5)
	{
		return interpolate(A, 0, 4, 255, 5);
	}
	if (A <= 0)
		return 255;
	if (A >= 5)
		return 255;
}

float lin_trunc(float a, float max)
{
	if (a>max)
	{
		return 1;
	}
	else return a / max;
}

sf::Color float_to_color_complex(float number_real, float number_complex, float min, float max)
{
	float R, G, B;
	float number = atan2(number_complex, number_real);
	float abs = sqrt(number_complex*number_complex + number_real*number_real);
	float grayscale = 2 * (number + 3.14159265) / (2 * 3.15159265) - 1;
	R = red_complex(grayscale, -1, 1);
	G = green_complex(grayscale, -1, 1);
	B = blue_complex(grayscale, -1, 1);
	float az = lin_trunc(abs, max), ak = asy(abs, min);
	R = az*R*(1 - ak) + 255 * ak;
	G = az*G*(1 - ak) + 255 * ak;
	B = az*B*(1 - ak) + 255 * ak;
	return sf::Color(R, G, B);
}

void updateTablej(float min, float max)
{
#pragma omp parallel for
	for (int i = 0; i < colortablesize; i++)
	{
		ColorTable[i] = float_to_colorj(min + (float)(max - min)*(float)i / (float)colortablesize, min, max);
	}
	def_min = min;
	def_max = max;
}


Color float2color(float number)
{
	if (number < def_min)
	{
		float k = def_min - number;
		k *= 2;
		k = (k*k - 1) / (1 + k*k) + 1;
		return Color(255, 128 * k, 100 * k);
	}
	else if (number > def_max)
	{
		float  k = number - def_max;
		k *= 2;
		k = (k*k - 1) / (1 + k*k) + 1;
		return Color(100 * k, 128 * k, 255);
	}
	else
	{
		int p = colortablesize * (number - def_min) / ((def_max - def_min));
		return ColorTable[p];
	}

}

Color float2colorj(float number)
{
	if (number < def_min)
	{
		return Color(0, 0, 0);
	}
	else if (number > def_max)
	{
		return Color(255, 255, 255);
	}
	else
	{
		int p = colortablesize * (number - def_min) / ((def_max - def_min));
		return ColorTable[p];
	}

}


void CPU_COLOR_GRAD(sf::Image* img, float** DATA, int dx, int dy)
{
	for (int X = 0; X<dx; X++)
#pragma omp parallel for
		for (int Y = 0; Y<dy; Y++)
		{
			Color c = float2color(DATA[X][Y]);
			img->setPixel(X, Y, c);
		}
}

void CPU_COLOR_GRAD_JET(sf::Image* img, float** DATA, int dx, int dy)
{
	for (int X = 0; X<dx; X++)
#pragma omp parallel for
		for (int Y = 0; Y<dy; Y++)
		{
			Color c = float2colorj(DATA[X][Y]);
			img->setPixel(X, Y, c);
		}
}

void CPU_COLOR_GRAD_COMPLEX(sf::Image* img, float** DATAR, float** DATAI, int dx, int dy)
{
	for (int X = 0; X<dx; X++)
#pragma omp parallel for
		for (int Y = 0; Y<dy; Y++)
		{
			Color c = float_to_color_complex(DATAR[X][Y], DATAI[X][Y], 0.6, 3);
			img->setPixel(X, Y, c);
		}
}

void CPU_COLOR_GRAD_2(sf::Image* img, float** DATA1, float** DATA2, int dx, int dy)
{
	for (int X = 0; X<dx; X++)
#pragma omp parallel for
		for (int Y = 0; Y<dy; Y++)
		{
			Color c1 = float2color(DATA1[X][Y]);
			Color c2 = float2color(DATA2[X][Y]);
			img->setPixel(X, Y, Color(c1.b, 0, c2.b));
		}
}

void CPU_COLOR_GRAD_3(sf::Image* img, float** DATA1, float** DATA2, int dx, int dy)
{
	for (int X = 0; X<dx; X++)
#pragma omp parallel for
		for (int Y = 0; Y<dy; Y++)
		{
			Color c1 = float2color(DATA1[X][Y]);
			Color c2 = float2color(std::abs(DATA2[X][Y]));
			img->setPixel(X, Y, Color(c1.r, c2.b, c1.b));
		}
}


class SFMLP
{
private:
	int width, height;
	RenderWindow window;
	vector<VertexArray> Lines;
	vector< vector<float> > Lines_x, Lines_y;
	vector<sf::Color> Lines_color;
	vector<string> Lines_legend;
	sf::Vector2i mouse_prev;
	int frame, touch;
	sf::Vertex Yaxis[2], Xaxis[2], OneX[2], OneY[2];
	float l_x, l_y, w_x, w_y;
	
	bool imageset;
	int imgX, imgY;
	float imgx0, imgy0, imgx1, imgy1;
	Texture texture;
	sf::Image mainimg;
	Sprite MainSprite;
	sf::Text su;

public:
	bool open;
	SFMLP(int w, int h, float WX, float WY, float lx, float ly) : width(w), height(h), w_x(WX), w_y(WY), l_x(lx), l_y(ly), frame(0), touch(-1), open(1), imageset(0)
	{
		window.create(sf::VideoMode(w, h), "Plot graph");
		window.setFramerateLimit(60);
		su.setFont(font);
		su.setCharacterSize(16);
	}

	~SFMLP()
	{
		if(window.isOpen()) window.close();
	}

	void SetImage(float**& A, int W, int H, float x0, float y0, float x1, float y1, float min, float max)
	{
		imageset = 1;
		imgx0 = x0;
		imgx1 = x1;
		imgy0 = y0;
		imgy1 = y1;

		imgX = W;
		imgY = H;

		texture.setSmooth(true);

		mainimg.create(W, H, Color::White);

		updateTablej(min, max);


		CPU_COLOR_GRAD_JET(&mainimg, A, W, H);

		texture.loadFromImage(mainimg);
		MainSprite.setTexture(texture);
	}

	void AddLine(vector<float> X, vector<float> Y, sf::Color C, string S)
	{
		Lines.emplace_back();
		int n = Lines.size()-1;
		Lines[n].resize(X.size());
		Lines[n].setPrimitiveType(LinesStrip);
        for(int i = 0; i < X.size(); i++)
        {
            Lines[n][i].color = C;
        }
		Lines_legend.push_back(S);
		Lines_color.push_back(C);
		Lines_x.push_back(X);
		Lines_y.push_back(Y);
	}

	void AddEmptyLine(sf::Color C, string S)
	{
		Lines.emplace_back();
		int n = Lines.size() - 1;
		Lines[n].resize(0);
		Lines[n].setPrimitiveType(LinesStrip);
		Lines_legend.push_back(S);
		Lines_color.push_back(C);
		Lines_x.emplace_back();
		Lines_y.emplace_back();
	}

	void AddPointToLine(int l, float x, float y)
	{
		int ll = Lines_x.size();
		if (l < ll)
		{
			int n = Lines_x[l].size();
			Lines[l].resize(n + 1);
			Lines[l][n].color = Lines_color[l];
			Lines_x[l].push_back(x);
			Lines_y[l].push_back(y);
		}
	}

	void UpdateState()
	{
		window.clear(Color::Black);
		frame++;
		sf::Vector2i dmouse = Mouse::getPosition(window) - mouse_prev;
		mouse_prev = Mouse::getPosition(window);

		//the change in mouse position
		float mdx = dmouse.x * w_x / (float)width;
		float mdy = dmouse.y * w_y / (float)height;

	

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
		{
			if (touch != -1)
			{
				//update the plot position
				l_x -= mdx;
				l_y += mdy;
			}
			touch = 1;
		}
		else
		{
			touch = -1;
		}

		sf::Event event;
		while (window.pollEvent(event))
		{
			float mm = (float)event.mouseWheel.delta / 10.f;
			switch (event.type)
			{
			case sf::Event::Closed:
				window.close();
				open = 0;
				break;
			case sf::Event::MouseWheelMoved:
				w_x *= 1 - mm;
				w_y *= 1 - mm;
				break;
			default:
				break;
			}
		}

		if (Keyboard::isKeyPressed(Keyboard::K))
		{
			w_x *= 1.01;
			w_y *= 1.01;
		}
		if (Keyboard::isKeyPressed(Keyboard::L))
		{
			w_x *= 0.99;
			w_y *= 0.99;
		}

		//plot the image
		if (imageset)
		{
			MainSprite.setPosition((imgx0 - l_x)*width/ w_x + width*0.5, -(imgy1 - l_y)*height / w_y + height*0.5);
			MainSprite.setScale((float)width * (imgx1-imgx0)/(w_x*imgX), (float)height * (imgy1 - imgy0) / (w_y*imgY));
			window.draw(MainSprite);
		}

		//plot the grid
		DrawGrid(l_x - w_x*0.5, l_x + w_x*0.5, l_y - w_y*0.5, l_y + w_y*0.5, sf::Color::Color(128, 128, 128), sf::Color::Color(128, 128, 128));

		//plot the axes
		DrawAxis(l_x - w_x*0.5, l_x + w_x*0.5, l_y - w_y*0.5, l_y + w_y*0.5, sf::Color::White);

		//plot the lines
		for (int i = 0; i < Lines_x.size(); i++)
		{	
			for (int j = 0; j < Lines_x[i].size(); j++)
			{
				int x = (Lines_x[i][j] - l_x)*width / w_x + width*0.5;
				int y = -(Lines_y[i][j] - l_y)*height / w_y + height*0.5;
				Lines[i][j].position = Vector2f(x, y);
			}
			window.draw(Lines[i]);
		}

		window.display();
	}


	//attention: shitcode ahead

	void DrawAxis(float x0, float x1, float y0, float y1, Color C)
	{
		Yaxis[0].position = Vector2f(width / 2 - ((x1 + x0) / 2)*(double)width / (x1 - x0), 0);
		Yaxis[1].position = Vector2f(width / 2 - ((x1 + x0) / 2)*(double)width / (x1 - x0), height);
		Xaxis[0].position = Vector2f(0, height / 2 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		Xaxis[1].position = Vector2f(width, height / 2 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		Yaxis[0].color = C;
		Yaxis[1].color = C;
		Xaxis[0].color = C;
		Xaxis[1].color = C;
		OneY[0].position = Vector2f(width / 2 - ((x1 + x0) / 2 - 1)*(double)width / (x1 - x0), height / 2 - 5 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		OneY[1].position = Vector2f(width / 2 - ((x1 + x0) / 2 - 1)*(double)width / (x1 - x0), height / 2 + 5 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		OneX[0].position = Vector2f(width / 2 + 5 - ((x1 + x0) / 2)*(double)width / (x1 - x0), height / 2 + ((y0 + y1) / 2 - 1)*(double)height / (y1 - y0));
		OneX[1].position = Vector2f(width / 2 - 5 - ((x1 + x0) / 2)*(double)width / (x1 - x0), height / 2 + ((y0 + y1) / 2 - 1)*(double)height / (y1 - y0));
		OneX[0].color = C;
		OneX[1].color = C;
		OneY[0].color = C;
		OneY[1].color = C;
		window.draw(Xaxis, 2, sf::Lines);
		window.draw(Yaxis, 2, sf::Lines);
		window.draw(OneX, 2, sf::Lines);
		window.draw(OneY, 2, sf::Lines);
		Yaxis[0].position = Vector2f(width / 2 - ((x1 + x0) / 2)*(double)width / (x1 - x0), 0);
		Yaxis[1].position = Vector2f(width / 2 - ((x1 + x0) / 2)*(double)width / (x1 - x0) + 15, 15);
		Xaxis[0].position = Vector2f(width / 2 - ((x1 + x0) / 2)*(double)width / (x1 - x0), 0);
		Xaxis[1].position = Vector2f(width / 2 - ((x1 + x0) / 2)*(double)width / (x1 - x0) - 15, 15);
		window.draw(Xaxis, 2, sf::Lines);
		window.draw(Yaxis, 2, sf::Lines);
		Xaxis[0].position = Vector2f(width, height / 2 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		Xaxis[1].position = Vector2f(width - 15, height / 2 + 15 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		Yaxis[0].position = Vector2f(width, height / 2 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		Yaxis[1].position = Vector2f(width - 15, height / 2 - 15 + ((y0 + y1) / 2)*(double)height / (y1 - y0));
		window.draw(Xaxis, 2, sf::Lines);
		window.draw(Yaxis, 2, sf::Lines);
	}

	void DrawGrid(float x0, float x1, float y0, float y1, Color C, Color Te)
	{
		float dex = x1 - x0, dey = y1 - y0;
		float dx = expf(logf(10)*(float)floorf(log10f(dex)))*(float)width / (4 * dex);
		float dy = expf(logf(10)*(float)floorf(log10f(dey)))*(float)height / (4 * dey);
		dx = min(dx, dy);
		dy = dx;
		float nx = ((float)width) / dx, ny = ((float)height) / dy;
		float dxr = dex / nx, dyr = dey / ny;
		float sy = (y1 / dyr - floorf(y1 / dyr))*(float)height / ny, sx = (x0 / dxr - floorf(x0 / dxr))*(float)width / nx;
		Yaxis[0].color = C;
		Yaxis[1].color = C;
		su.setColor(Te);
		for (int i = 0; i<nx+1; i++)
		{
			Yaxis[0].position = Vector2f(-sx + dx*i, 0);
			Yaxis[1].position = Vector2f(-sx + dx*i, height);
			window.draw(Yaxis, 2, sf::Lines);
			su.setPosition(-sx + dx*i + 5, height / 2 + ((y0 + y1) / 2)*(double)height / dey + 5);
			su.setString(num2str((floorf(x0 / dxr) + i)*dxr));
			window.draw(su);
		}
		for (int i = 0; i<ny+1; i++)
		{
			Yaxis[0].position = Vector2f(0, sy + dy*i);
			Yaxis[1].position = Vector2f(width, sy + dy*i);
			window.draw(Yaxis, 2, sf::Lines);
			su.setPosition(width / 2 - ((x1 + x0) / 2)*(double)width / dex + 5, sy + dy*i + 5);
			su.setString(num2str((floorf(y1 / dyr) - i)*dyr));
			window.draw(su);
		}
	}

};