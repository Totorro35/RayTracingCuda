#pragma once
#include <vector>

#include "device_launch_parameters.h"
#include "Geometry/ray.cuh"
#include "Geometry/Triangle.cuh"
#include "Geometry/CastedRay.cuh"
#include "Geometry/PointLight.cuh"

class BVH {
	Triangle m_tris[500];
	int size;
	PointLight m_lights[8];
	int size_lights = 1;

public:

	__host__ BVH(const std::vector<Triangle>& tris){
		int i = 0;
		for (const Triangle & tri : tris) {
			++i;
			if (i == 500) {
				break;
			}
			m_tris[i] = tri;
		}
		size = i;
		std::cout << "NB_Tri :" << i << std::endl;
		m_lights[0] = PointLight(Math::makeVector(0.0f,0.0f,4.95f),RGBColor(1.0f,1.0f,1.0f)*10);
	}

	__device__ __host__ bool intersect(Ray const & ray, RayTriangleIntersection & intersect) const {
		CastedRay cray(ray);
		for (int i = 0; i < size;++i) {
			cray.intersect(&m_tris[i]);
		}
		bool result = cray.validIntersectionFound();
		if (result) {
			intersect = cray.intersectionFound();
		}
		return result;
	}

	__device__ __host__ const PointLight* getLights() const {
		return m_lights;
	}

	__device__ __host__ int getNumberLights() const {
		return size_lights;
	}

};