#ifndef _Geometry_Light_H
#define _Geometry_Light_H

#include <Geometry/Material.cuh>
#include "Geometry/RayTriangleIntersection.cuh"


	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \class	PointLight
	///
	/// \brief	A point light.
	///
	/// \author	F. Lamarche, Université de Rennes 1
	/// \date	04/12/2013
	////////////////////////////////////////////////////////////////////////////////////////////////////
	class PointLight
	{
	protected:
		/// \brief	The light position.
		Math::Vector3f m_position ;

		/// \brief	The light color.
		RGBColor m_color ;

		float m_radius;

		double m_score;

	public:

		__device__ __host__ PointLight(Math::Vector3f const & position = Math::makeVector(0.0,0.0,0.0), RGBColor const & color = RGBColor(),float radius=1.0f)
			: m_position(position), m_color(color), m_radius(radius)
		{}

		__device__ __host__  const RGBColor & color() const
		{ return m_color ; }

		__device__ __host__  void computeScore() {
			//m_score = m_color.grey()*4*M_PI*m_radius*m_radius;
		}

		__device__ __host__  double getScore() const {
			return m_score;
		}

		__device__ __host__  const Math::Vector3f & position() const
		{ return m_position ; }

		__device__ __host__ float getRadius() const {
			return m_radius;
		}

	} ;

#endif
