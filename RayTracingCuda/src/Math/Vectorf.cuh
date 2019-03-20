#ifndef _HelperGL_VectorGl_H
#define _HelperGL_VectorGl_H

#include <Math/Vector.cuh>

namespace Math
{
	typedef Vector<float,2> Vector2f ;
	typedef Vector<float,3> Vector3f ;
	typedef Vector<float,4> Vector4f ;

	__device__ __host__ static Math::Vector3f getVector(double theta, double phy)
	{
		return Math::makeVector(sin(theta)*cos(phy), sin(theta)*sin(phy), cos(theta));
	}
}

#endif