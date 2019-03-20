#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Math/Vectorf.cuh"

	class Ray
	{
		
	public:
		Math::Vector3f m_source;

	protected:
		Math::Vector3f m_direction;
		Math::Vector3f m_inv;
		int m_sign[3];

	public:


		//I do not check if the direction is normalized
		__device__ __host__ Ray(Math::Vector3f const& p_source, Math::Vector3f const& p_dir) :
			m_source(p_source),
			m_direction(p_dir.normalized())
		{
			m_inv = Math::makeVector(1.0f / m_direction[0], 1.0f / m_direction[1], 1.0f / m_direction[2]);
			m_sign[0] = m_direction[0] < 0.0;
			m_sign[1] = m_direction[1] < 0.0;
			m_sign[2] = m_direction[2] < 0.0;
		}


		__device__ __host__ Math::Vector3f const& source()const {
			return m_source;
		}


		__device__ __host__ Math::Vector3f const& direction()const
		{
			return m_direction;
		}

		__device__ __host__ Math::Vector3f const& inv_direction()const
		{
			return m_inv;
		}

		__device__ __host__ const int * getSign() const
		{
			return m_sign;
		}

		__device__ __host__ void project(Math::Vector3f const & point, float & t, Math::Vector3f & delta)
		{
			t = (point - source())*direction();
			delta = (point - source()) - direction()*t;
		}

	};
