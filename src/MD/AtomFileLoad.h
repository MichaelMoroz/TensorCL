#pragma once
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>


namespace fs = std::filesystem;


float random()
{
	return (rand() / (float)(RAND_MAX - 1));
}


/*
//box-muller gausian random numbers
float randomn(float d)
{
	return d * sqrt(2 * abs(log(random() + 1e-2)))*cos(2 * 3.14159*random());
}
*/

float randd(float a, float b)
{
	return random()*(b - a) + a;
}

float randomd(float a)
{
	return 2 * a*(random() - 0.5);
}

struct float3
{
	float x;
	float y;
	float z;

	float3()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	float3(std::vector<float> a)
	{
		x = a[0];
		y = a[1];
		z = a[2];
	}



	void operator=(float3 b)
	{
		(*this).x = b.x;
		(*this).y = b.y;
		(*this).z = b.z;
	}

	float& operator[](int b)
	{
		switch (b)
		{
		case 0: return x;
		case 1: return y;
		case 2: return z;
		}
	}
};

struct mat3
{
	float a[3][3];
};

float3 rrd()
{
	float3 a;
	a.x = randd(-0.5, 0.5);
	a.y = randd(-0.5, 0.5);
	a.z = randd(-0.5, 0.5);
	return a;
}

std::vector<float> get_vec(float3 a)
{
	std::vector<float> out;
	out.push_back(a.x);
	out.push_back(a.y);
	out.push_back(a.z);
	return out;
}



float3 operator+(float3 a, float3 b)
{
	float3 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	c.z = a.z + b.z;
	return c;
}

float3 operator-(float3 a, float3 b)
{
	float3 c;
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	c.z = a.z - b.z;
	return c;
}

float operator*(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}



float3 operator*(float3 a, float b)
{
	float3 c;
	c.x = a.x*b;
	c.y = a.y*b;
	c.z = a.z*b;
	return c;
}

float3 operator*(float b, float3 a)
{
	float3 c;
	c.x = a.x*b;
	c.y = a.y*b;
	c.z = a.z*b;
	return c;
}

//matrix vector multiplication
float3 operator*(mat3 A, float3 B)
{
	float3 C;
	C.x = A.a[0][0] * B.x + A.a[0][1] * B.y + A.a[0][2] * B.z;
	C.y = A.a[1][0] * B.x + A.a[1][1] * B.y + A.a[1][2] * B.z;
	C.z = A.a[2][0] * B.x + A.a[2][1] * B.y + A.a[2][2] * B.z;
	return C;
}

float length(float3 a)
{
	return sqrt(a*a);
}

float3 normalize(float3 a)
{
	return a * (1 / length(a));
}

float3 rrdn()
{
	float3 a;
	a.x = randd(-0.5, 0.5);
	a.y = randd(-0.5, 0.5);
	a.z = randd(-0.5, 0.5);
	return a * (1 / length(a));
}

mat3 RotMat(float angle, float3 axis)
{
	mat3 rotationMatrix;
	float u = axis.x, v = axis.y, w = axis.z;
	float L = (u*u + v * v + w * w);
	angle = angle * 3.14159 / 180.0; //converting to radian value
	float u2 = u * u;
	float v2 = v * v;
	float w2 = w * w;

	rotationMatrix.a[0][0] = (u2 + (v2 + w2) * cos(angle)) / L;
	rotationMatrix.a[0][1] = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
	rotationMatrix.a[0][2] = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;

	rotationMatrix.a[1][0] = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
	rotationMatrix.a[1][1] = (v2 + (u2 + w2) * cos(angle)) / L;
	rotationMatrix.a[1][2] = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;

	rotationMatrix.a[2][0] = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
	rotationMatrix.a[2][1] = (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
	rotationMatrix.a[2][2] = (w2 + (u2 + v2) * cos(angle)) / L;

	return rotationMatrix;
}

float3 rotate(mat3 rm, float3 x)
{
	return rm * x;
}


//atom charges
float a2c(std::string a)
{
	if (a == "H")
	{
		return 1;
	}
	else if (a == "He")
	{
		return 2;
	}
	else if (a == "Li")
	{
		return 3;
	}
	else if (a == "Be")
	{
		return 4;
	}
	else if (a == "B")
	{
		return 5;
	}
	else if (a == "C")
	{
		return 6;
	}
	else if (a == "N")
	{
		return 7;
	}
	else if (a == "O")
	{
		return 8;
	}
	else if (a == "F")
	{
		return 9;
	}
	else if (a == "Ne")
	{
		return 10;
	}
	else if (a == "Na")
	{
		return 11;
	}
	else if (a == "Mg")
	{
		return 12;
	}
	else if (a == "Al")
	{
		return 13;
	}
	else if (a == "Si")
	{
		return 14;
	}
	else if (a == "P")
	{
		return 15;
	}
	else if (a == "S")
	{
		return 16;
	}
	else if (a == "Cl")
	{
		return 17;
	}
	else if (a == "Ar")
	{
		return 18;
	}
	else if (a == "K")
	{
		return 19;
	}
	else if (a == "Ca")
	{
		return 20;
	}
	else if (a == "Sc")
	{
		return 21;
	}
	else if (a == "Ti")
	{
		return 22;
	}
	else if (a == "V")
	{
		return 23;
	}
	else if (a == "Cr")
	{
		return 24;
	}
	else if (a == "Mn")
	{
		return 25;
	}
	else if (a == "Fe")
	{
		return 26;
	}
	else if (a == "Co")
	{
		return 27;
	}
	else if (a == "Ni")
	{
		return 28;
	}
	else if (a == "Cu")
	{
		return 29;
	}
	else if (a == "Zn")
	{
		return 30;
	}
	else if (a == "Ga")
	{
		return 31;
	}
	else
	{
		return 0;
	}
}

std::string c2a(int a)
{
	if (a == 8)
	{
		return "O";
	}
	else if (a == 30)
	{
		return "Zn";
	}
	else
	{
		return "N//A";
	}
}

float c2m(int a)
{
	if (a == 8)
	{
		return 15.999f;
	}
	else if (a == 30)
	{
		return 65.38f;
	}
	else
	{
		return 1;
	}
}

struct Cluster
{
	float Energy, BindingEnergy, HOMO, LUMO, SmearingEnergy;
	std::vector< float > atom_id;
	std::vector< std::vector<float> > atom_coords;
};

//get path vector of all files of given type
std::vector<fs::path> GetFilesInFolder(std::string folder, std::string filetype)
{
	std::vector<fs::path> paths;

	for (const auto & entry : fs::directory_iterator(folder))
	{
		//check if the file has the correct filetype
		if (entry.path().extension().string() == filetype)
		{
			paths.push_back(entry.path());
		}
	}

	return paths;
}

Cluster XYZ_Load(std::string filename)
{
	Cluster DATA;
	std::ifstream file(filename);
	std::string line;
	int j = 0;
	int N = 0;
	while (getline(file, line))
	{
		j++;
		int i = 0;

		std::istringstream iss(line);
		//atom num
		if (j == 1)
		{
			iss >> N;
		}
		//energy and stuff
		else if (j == 2)
		{	
			iss >> DATA.Energy;
			iss >> DATA.BindingEnergy;
			iss >> DATA.HOMO;
			iss >> DATA.LUMO;
			iss >> DATA.SmearingEnergy;
		}
		//atom coordinates and type
		else
		{
			std::string Atom; iss >> Atom;
			float charge = a2c(Atom); //get atom charge
			DATA.atom_id.push_back(charge);

			std::vector<float> cxyz; float num;
			for (int k = 0; k < 3; k++)
			{
				iss >> num;
				cxyz.push_back(num);
			}
			DATA.atom_coords.push_back(cxyz);
		}
	}
	return DATA;
}

Cluster RandomRotateCluster(Cluster CC)
{
	int N = CC.atom_coords.size();
	mat3 rotmat = RotMat(3.14159f*randd(-1, 1), rrd());
	for (auto i = 0; i < N; i++)
	{
		float3 X(CC.atom_coords[i]);
		CC.atom_coords[i] = get_vec(rotmat*X);
	}
	return CC;
}