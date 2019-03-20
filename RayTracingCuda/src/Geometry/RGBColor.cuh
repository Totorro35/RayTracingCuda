#ifndef _Geometry_RGBColor
#define _Geometry_RGBColor

#include <iostream>
#include <algorithm>
#include <cassert>

	class RGBColor
	{
	protected:
		/// \brief	The red(0), green(1) and blue (2) components.
		float m_color[3];

	public:

		__device__ __host__ RGBColor(float R = 0, float G = 0, float B = 0)
		{
			m_color[0] = R;
			m_color[1] = G;
			m_color[2] = B;
			//validateColor();
		}

		__device__ __host__ bool isBlack() const
		{
			return m_color[0] == 0.0f && m_color[1] == 0.0f && m_color[2] == 0.0f;
		}

		__device__ __host__ void set(float r, float g, float b)
		{
			m_color[0] = r;
			m_color[1] = g;
			m_color[2] = b;
		}

		__device__ __host__ void set(float * rgb)
		{
			m_color[0] = rgb[0];
			m_color[1] = rgb[1];
			m_color[2] = rgb[2];
		}

		__device__ __host__ float grey() const
		{
			return (m_color[0] + m_color[1] + m_color[2]) / 3.0;
		}

		__device__ __host__ RGBColor operator+ (RGBColor const & c) const
		{
			return RGBColor(c.m_color[0] + m_color[0], c.m_color[1] + m_color[1], c.m_color[2] + m_color[2]);
		}

		__device__ __host__ RGBColor operator* (RGBColor const & c) const
		{
			return RGBColor(c.m_color[0] * m_color[0], c.m_color[1] * m_color[1], c.m_color[2] * m_color[2]);
		}

		__device__ __host__ RGBColor toneMappe() const
		{
			return RGBColor(m_color[0] /(m_color[0]+1), m_color[1] / (m_color[1] + 1), m_color[2] / (m_color[2] + 1))*256;
		}

		__device__ __host__ RGBColor operator* (float v) const
		{
			return RGBColor(m_color[0] * v, m_color[1] * v, m_color[2] * v);
		}

		__device__ __host__ RGBColor operator/ (float v) const
		{
			return RGBColor(m_color[0] / v, m_color[1] / v, m_color[2] / v);
		}

		__device__ __host__ float operator[] (int c) const
		{
			return m_color[c];
		}

		__device__ __host__ float & operator[] (int c)
		{
			return m_color[c];
		}

		__device__ __host__ bool operator==(RGBColor const & color) const
		{
			return m_color[0] == color[0] && m_color[1] == color[1] && m_color[2] == color[2];
		}

		__device__ __host__ bool operator!=(RGBColor const & color) const
		{
			return !((*this) == color);
		}

		__host__ bool operator<(const RGBColor & color) const
		{
			return std::lexicographical_compare(m_color, m_color + 3, color.m_color, color.m_color + 3);
		}

	};

	inline ::std::ostream & operator << (::std::ostream & out, RGBColor const & color)
	{
		out << "(" << color[0] << "," << color[1] << "," << color[2] << ")";
		return out;
	}
#endif
