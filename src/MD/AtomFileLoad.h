#pragma once
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>


using namespace std;
namespace fs = std::filesystem;

//atom charges
float a2c(string a)
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

string c2a(int a)
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

//get path vector of all files of given type
vector<fs::path> GetFilesInFolder(string folder, string filetype)
{
	vector<fs::path> paths;

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


vector<vector<float>> XYZ_ForceEnergy_Load(string filename)
{
	vector<vector<float>> c_xyz_grad_e;
	vector<vector<float>> c_xyz;
	vector<vector<float>> grad_e;
	ifstream file(filename.c_str());
	string line; 
	int j = 0;
	int N = 0;
	float E = 0;
	while (getline(file, line))
	{
		j++;
		int i = 0;
		if (j == 2)
		{
			std::replace(line.begin(), line.end(), ';', ' ');
			std::replace(line.begin(), line.end(), '[', ' ');
			std::replace(line.begin(), line.end(), ',', ' ');
			std::replace(line.begin(), line.end(), ']', ' ');
		}
			
		istringstream iss(line);
		//atom num
		if (j == 1)
		{
			iss >> N;
		}
		//atom configuration energy and forces
		else if (j == 2)
		{
			iss >> E;
		
			for (int k = 0; k < N; k++)
			{	
				vector<float> G;
				for (int p = 0; p < 3; p++)
				{
					float num=0;
					iss >> num;
					G.push_back(num*kcalmol_in_eV);
				}
				G.push_back(E*kcalmol_in_eV);
				grad_e.push_back(G);
			}	
		}
		//atom coordinates and type
		else
		{
			string Atom;
			iss >> Atom;
			float charge = a2c(Atom); //get atom charge
			float num;
			vector<float> cxyz;
			cxyz.push_back(charge);
			for (int k = 0; k < 3; k++)
			{
				iss >> num;
				cxyz.push_back(num);
			}
			c_xyz.push_back(cxyz);
		}
	}

	for (int i = 0; i < c_xyz.size(); i++)
	{
		std::vector<float> AB = c_xyz[i];
		AB.insert(AB.end(), grad_e[i].begin(), grad_e[i].end());

		c_xyz_grad_e.push_back(AB);
	}

	return c_xyz_grad_e;
}

vector<vector<float>> XYZF_Load(string filename)
{
	vector<vector<float>> c_xyz_grad_e;
	vector<vector<float>> c_xyz;
	vector<vector<float>> grad_e;
	ifstream file(filename.c_str());
	string line;
	int j = 0;
	int N = 0;
	float E = 0;
	while (getline(file, line))
	{
		j++;
		int i = 0;

		istringstream iss(line);
		//atom num
		if (j == 1)
		{
			iss >> N;
		}
		//??
		else if (j == 2)
		{
		}
		//atom coordinates and type
		else
		{
			string Atom;
			iss >> Atom;
			float charge = a2c(Atom); //get atom charge
			float num;
			vector<float> cxyz, grade;
			cxyz.push_back(charge);
			for (int k = 0; k < 3; k++)
			{
				iss >> num;
				cxyz.push_back(num);
			}
			for (int k = 0; k < 3; k++)
			{
				iss >> num;
				grade.push_back(num);
			}
			grade.push_back(E);
			c_xyz.push_back(cxyz);
			grad_e.push_back(grade);
		}
	}

	for (int i = 0; i < c_xyz.size(); i++)
	{
		std::vector<float> AB = c_xyz[i];
		AB.insert(AB.end(), grad_e[i].begin(), grad_e[i].end());

		c_xyz_grad_e.push_back(AB);
	}

	return c_xyz_grad_e;
}


vector<vector<float>> XYZ_Load(string filename)
{
	vector<vector<float>> c_xyz;
	ifstream file(filename.c_str());
	string line;
	int j = 0;
	int N = 0;
	float3 center;
	while (getline(file, line))
	{
		j++;
		int i = 0;

		istringstream iss(line);
		//atom num
		if (j == 1)
		{
			iss >> N;
		}
		//??
		else if (j == 2)
		{
		}
		//atom coordinates and type
		else
		{
			string Atom;
			iss >> Atom;
			float charge = a2c(Atom); //get atom charge
			float num;
			vector<float> cxyz;
			cxyz.push_back(charge);
			for (int k = 0; k < 3; k++)
			{
				iss >> num;
				cxyz.push_back(num);
			}
			c_xyz.push_back(cxyz);
		}
	}

	//set the center at the center of mass
	for (int i = 0; i < N; i++)
	{
		float3 ac;
		ac[0] = c_xyz[i][1];
		ac[1] = c_xyz[i][2];
		ac[2] = c_xyz[i][3];
		center = center + ac;
	}

	center = center*(1.f / N);

	for (int i = 0; i < N; i++)
	{
		c_xyz[i][1] -= center[0] + 0*randd(-0.3,0.3);
		c_xyz[i][2] -= center[1] + 0*randd(-0.3, 0.3);
		c_xyz[i][3] -= center[2] + 0*randd(-0.3, 0.3);
	}

	return c_xyz;
}