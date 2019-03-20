#ifndef _Geometry_Texture_H
#define _Geometry_Texture_H

#include <string>
//#include <SOIL.h>
#include <Geometry/RGBColor.cuh>
#include <Math/Vectorf.cuh>

	class Texture
	{
	private:
		int m_width;
		int m_height;
		unsigned char * m_data;

	public:
		Texture(const ::std::string & filename)
		{
			//m_data = SOIL_load_image(filename.c_str(), &m_width, &m_height, 0, SOIL_LOAD_RGB);
			if (m_data == NULL)
			{
				::std::cout << "Invalid texture file: " << filename << ::std::endl;
			}
		}

		~Texture()
		{
			//SOIL_free_image_data(m_data);
		}

		__device__ __host__ bool isValid() const
		{
			return m_data != NULL;
		}

		__device__ __host__ Math::Vector2f getSize() {
			Math::Vector2f result;
			result[0] = float(m_width);
			result[1] = float(m_height);
			return result;
		}

		__device__ __host__ RGBColor pixel(int x, int y) const
		{
			while (x < 0) { x += m_width; }
			while (y < 0) { y += m_height; }
			//std::cout << m_width << " ; " << m_height << std::endl;
			x = x%m_width;
			
			y = y%m_height;
			
			int offset = y * 3 * m_width + x * 3;
			unsigned char r = m_data[offset];
			unsigned char g = m_data[offset + 1];
			unsigned char b = m_data[offset + 2];
			return RGBColor(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f);
		}

		__device__ __host__ RGBColor pixel(Math::Vector2f const & v) const
		{
			return pixel(int(v[0] * m_width), int(v[1] * m_height));
		}
	};

#endif