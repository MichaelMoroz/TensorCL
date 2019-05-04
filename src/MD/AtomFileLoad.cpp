#include "AtomFileLoad.h"

float3::float3()
{
	x = 0;
	y = 0;
	z = 0;
}

float3::float3(vector<float> a)
{
	x = a[0];
	y = a[1];
	z = a[2];
}
