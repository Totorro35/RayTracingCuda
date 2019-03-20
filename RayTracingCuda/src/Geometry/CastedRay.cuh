#ifndef _Geometry_CastedRay
#define _Geometry_CastedRay

#include <assert.h>
#include <Geometry/ray.cuh>
#include <Geometry/Triangle.cuh>
#include <Geometry/RayTriangleIntersection.cuh>

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \class	CastedRay
	///
	/// \brief	A ray that keeps track of the nearest intersection found so far.
	///
	/// \author	F. Lamarche, Universit� de Rennes 1
	/// \date	04/12/2013
	////////////////////////////////////////////////////////////////////////////////////////////////////
	class CastedRay : public Ray
	{
	public:
	protected:
		/// \brief	The nearest intersection found so far.
		RayTriangleIntersection m_intersection ;

	public:

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	CastedRay::CastedRay(Math::Vector3 const & source, Math::Vector3 const & direction)
		///
		/// \brief	Constructor.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	source   	The ray source.
		/// \param	direction	The ray direction.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ CastedRay(Math::Vector3f const & source, Math::Vector3f const & direction)
			: Ray(source, direction)
		{}

		__device__ __host__ CastedRay(Ray const & ray)
			: Ray(ray)
		{}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	bool CastedRay::intersect(Triangle * triangle)
		///
		/// \brief	Computes the intersection between the current ray and the provided triangle. If the coputed 
		/// 		intersection is the nearest to the source, it is recorded.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param [in,out]	triangle	The triangle.
		///
		/// \return	true if it succeeds, false if it fails.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ bool intersect(const Triangle * triangle)
		{
			RayTriangleIntersection intersection(triangle, *this) ;
			if(intersection<m_intersection)
			{
				m_intersection=intersection;
			}
			return intersection.valid() ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	RayTriangleIntersection const & CastedRay::intersectionFound() const
		///
		/// \brief Returns the nearest ntersection found so far.
		/// 	   
		/// \warning A call to this method should be peceeded by a call to validIntersectionFound to ensure
		/// 		 that the returned intersection is valid.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \return	.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ RayTriangleIntersection const & intersectionFound() const
		{ return m_intersection ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	bool CastedRay::validIntersectionFound() const
		///
		/// \brief	To know if a valid intersection as been found so far.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \return	true if it succeeds, false if it fails.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ bool validIntersectionFound() const
		{ return m_intersection.valid() ; }
	};

#endif
