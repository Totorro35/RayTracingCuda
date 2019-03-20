#ifndef _Geometry_RayTriangleIntersection_H
#define _Geometry_RayTriangleIntersection_H

#include <Geometry/ray.cuh>
#include <Geometry/Triangle.cuh>
#include <assert.h>

	class RayTriangleIntersection
	{
	protected:
		/// \brief	The distance between ray source and the intersection.
		float m_t ;
		/// \brief	The u coordinate of the intersection.
		float m_u ;
		/// \brief	The v coordinate of the intersection.
		float m_v ;
		/// \brief	Is the intersection valid?
		bool m_valid ;
		/// \brief	The triangle associated to the intersection.
		const Triangle * m_triangle ;

	public:
		__device__ __host__ RayTriangleIntersection(const Triangle * triangle, const Ray & ray)
			: m_triangle(triangle)
		{
			m_valid=triangle->intersection(ray, m_t, m_u, m_v) ;
		}

		__device__ __host__ RayTriangleIntersection()
			: m_valid(false), m_triangle(NULL)
		{}

		__device__ __host__ bool valid() const
		{ return m_valid ; }

		__device__ __host__ float tRayValue() const
		{ return m_t ; }

		__device__ __host__ float uTriangleValue() const
		{ return m_u ; }

		__device__ __host__ float vTriangleValue() const
		{ return m_v ; }

		__device__ __host__ const Triangle * triangle() const
		{ return m_triangle ; }

		__device__ __host__ Math::Vector3f intersection() const
		{
			return m_triangle->samplePoint(m_u, m_v);
		}

		__device__ __host__ bool operator < (RayTriangleIntersection const & i) const
		{ 
			return (m_valid && i.m_valid && (m_t<i.m_t)) || (!i.m_valid) ; 
		}
	} ;

#endif
