#pragma once
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <filesystem>


namespace fs = std::filesystem;


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

float3 operator*(mat3 A, float3 B);

struct Cluster
{
	float Energy, BindingEnergy, HOMO, LUMO, SmearingEnergy;
	std::vector< float > atom_id;
	std::vector< std::vector<float> > atom_coords;
};

//get path vector of all files of given type
std::vector<fs::path> GetFilesInFolder(std::string folder, std::string filetype);

Cluster XYZ_Load(std::string filename);

Cluster RandomRotateCluster(Cluster CC);