#ifndef _Geometry_camera_H
#define _Geometry_camera_H

#include <Math/Vectorf.cuh>
#include <Math/Quaternion.cuh>
#include <Geometry/ray.cuh>
#include <math.h>

	class Camera
	{
	protected:
		/// \brief	The camera position.
		Math::Vector3f m_position ;
		/// \brief	The aim of the camera.
		Math::Vector3f m_target ;
		/// \brief	Distance of the focal plane.
		float		  m_planeDistance ;
		/// \brief	Width of the projection rectangle.
		float	      m_planeWidth ;
		/// \brief	Height of the projection rectangle.
		float		  m_planeHeight ;
		/// \brief	The front vector of the camera.
		Math::Vector3f m_front ;
		/// \brief	The right vector.
		Math::Vector3f m_right ;
		/// \brief	The down vector.
		Math::Vector3f m_down ;
		/// \brief	The width vector of the projection rectangle.
		Math::Vector3f m_widthVector ;
		/// \brief	The height vector of the projection rectangle.
		Math::Vector3f m_heightVector ;
		/// \brief	The upper left point oft he projection rectangle.
		Math::Vector3f m_upLeftPoint ;

		__device__ __host__ void computeParameters()
		{
			m_front = m_target-m_position ;
			m_front = m_front/m_front.norm() ;
			m_right = Math::Quaternion<float>(Math::makeVector(0.0f, 0.0f, 1.0f), -3.14159265f/2.0f).rotate(m_front) ;
			m_right[2] = 0.0f;
			m_right = m_right/m_right.norm() ;
			m_down  = m_front^m_right ;
			m_down  = m_down/m_down.norm() ;
			m_widthVector  = m_right*m_planeWidth ;
			m_heightVector = m_down*m_planeHeight ;
			m_upLeftPoint  = m_position+m_front*m_planeDistance-m_widthVector*0.5-m_heightVector*0.5 ;
		}

	public:

		__device__ __host__ const Math::Vector3f & getFront() {
			return m_front;
		}

		__device__ __host__ const Math::Vector3f & getRight() {
			return m_right;
		}

		__device__ __host__ const Math::Vector3f & getUp() {
			return -m_down;
		}

		__device__ __host__ Camera(Math::Vector3f const & position = Math::makeVector(0.0f, 0.0f, 0.0f), 
			   Math::Vector3f const & target = Math::makeVector(0.0f, 1.0f, 0.0f), 
			float planeDistance=1.0f, float planeWidth=1.0f, float planeHeight=1.0f)
		   : m_position(position), m_target(target), m_planeDistance(planeDistance), 
		     m_planeWidth(planeWidth), m_planeHeight(planeHeight)
		{
			computeParameters() ;
		}

		__device__ __host__ void translateLocal(Math::Vector3f const & translation)
		{
			Math::Vector3f trans =m_right*translation[0] + m_front*translation[1] - m_down*translation[2];
			m_position = m_position + trans;
			m_target = m_target + trans;
			computeParameters();
		}

		__device__ __host__ void setPosition(Math::Vector3f const & position)
		{
			m_position = position ;
			computeParameters() ;
		}

		__device__ __host__ void setTarget(Math::Vector3f const & target)
		{
			m_target = target ;
			computeParameters() ;
		}

		__device__ __host__ void orienter(float xRel, float yRel) {

			Math::Vector3f orientation = m_target - m_position;
			orientation.normalized();
			orientation[0] = orientation[0] + xRel;
			orientation[2] = orientation[2] + yRel;
			orientation.normalized();
			Math::Vector3f result = m_position + orientation;
			setTarget(result);
		}

		inline __device__ __host__ Math::Vector3f const & getTarget()
		{
			return m_target;
		}

		inline __device__ __host__  Math::Vector3f const & getPosition()
		{
			return m_position;
		}

		__device__ Ray getRay(float coordX, float coordY) const
		{
			return Ray(m_position, m_upLeftPoint+m_widthVector*coordX+m_heightVector*coordY-m_position) ;
		}
	} ;

#endif
