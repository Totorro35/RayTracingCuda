#ifndef _Geometry_Triangle
#define _Geometry_Triangle

#include <Math/Vectorf.cuh>
#include <Geometry/Ray.cuh>
#include <Geometry/Material.cuh>
#include <cassert>

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// \class	Triangle
	///
	/// \brief	A triangle.
	///
	/// \author	F. Lamarche, Universit� de Rennes 1
	/// \date	04/12/2013
	////////////////////////////////////////////////////////////////////////////////////////////////////
	class Triangle
	{
	protected:
		/// \brief	Pointers to the three vertices
		Math::Vector3f * m_vertex[3] ; 
		/// \brief Pointers to the texture coordinates
		Math::Vector2f * m_textureCoordinates[3];
		/// \brief	The vertex 0 (added to enhance cache consistency)
		Math::Vector3f m_vertex0 ;
		/// \brief	The u axis.
		Math::Vector3f m_uAxis ; 
		/// \brief	The v axis.
		Math::Vector3f m_vAxis ;
		/// \brief	The normal.
		Math::Vector3f m_normal ;
		/// \brief	The associated material.
		Material * m_material ;
		/// \brief	Per vertex normal.
		Math::Vector3f m_vertexNormal[3];

	public:


		/// <summary>
		/// Sets a vertex normal.
		/// </summary>
		/// <param name="index">The index of the vertex.</param>
		/// <param name="normal">The normal.</param>
		__device__ __host__ void setVertexNormal(unsigned int index, Math::Vector3f const & normal)
		{
			m_vertexNormal[index] = normal;
		}


		/// <summary>
		/// Gets a vertex normal.
		/// </summary>
		/// <param name="index">The index of the vertex.</param>
		/// <returns></returns>
		__device__ __host__ const Math::Vector3f & getVertexNormal(unsigned int index) const
		{
			return m_vertexNormal[index];
		}

		/// <summary>
		/// Gets the vertex normal oriented toward a given point.
		/// </summary>
		/// <param name="index">The index of the vertex.</param>
		/// <param name="toward">The point.</param>
		/// <returns></returns>
		__device__ __host__ Math::Vector3f getVertexNormal(unsigned int index, const Math::Vector3f & toward) const
		{
			const Math::Vector3f & normal = m_vertexNormal[index];
			if ((toward - vertex(index))*normal < 0.0)
			{
				return -normal;
			}
			return normal;
		}

		/// <summary>
		/// Gets a pointer to the vertex normals array (size = 3, one normal per vertex).
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const Math::Vector3f * getVertexNormals() const
		{
			return m_vertexNormal;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	void Triangle::update()
		///
		/// \brief	Updates precomputed data. This method should be called if vertices are externally 
		/// 		modified.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ void update()
		{
			m_vertex0 = *m_vertex[0] ;
			m_uAxis = (*m_vertex[1])-(*m_vertex[0]) ;
			m_vAxis = (*m_vertex[2])-(*m_vertex[0]) ;
			m_normal = m_uAxis^m_vAxis ;
			m_normal = m_normal*(1.0f/m_normal.norm()) ;
			m_vertexNormal[0] = m_normal;
			m_vertexNormal[1] = m_normal;
			m_vertexNormal[2] = m_normal;
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="Triangle"/> class.
		/// </summary>
		/// <param name="a">A pointer to the first vertex.</param>
		/// <param name="b">A pointer to the second vertex.</param>
		/// <param name="c">A pointer to the third vertex.</param>
		/// <param name="ta">The texture coordinates of the first vertex.</param>
		/// <param name="tb">The texture coordinates of the second vertex.</param>
		/// <param name="tc">The texture coordinates of the third vertex.</param>
		/// <param name="material">The material.</param>
		__host__ Triangle(Math::Vector3f * a, Math::Vector3f * b, Math::Vector3f * c,
				 Math::Vector2f * ta, Math::Vector2f * tb, Math::Vector2f * tc, Material * material, const Math::Vector3f * normals=nullptr)
		{
			m_vertex[0] = a;
			m_vertex[1] = b;
			m_vertex[2] = c;
			m_textureCoordinates[0] = ta;
			m_textureCoordinates[1] = tb;
			m_textureCoordinates[2] = tc;
			m_material = material;
			update();
			if (normals != nullptr)
			{
				::std::copy(normals, normals+3, m_vertexNormal);
			}
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Triangle::Triangle(Math::Vector3 * a, Math::Vector3 * b, Math::Vector3 * c,
		/// 	Material * material)
		///
		/// \brief	Constructor.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	a					A pointer to the first vertex.
		/// \param	b					A pointer to the second vertex.
		/// \param	c					A pointer to the third vertex.
		/// \param [in,out]	material	If non-null, the material.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ Triangle(Math::Vector3f * a, Math::Vector3f * b, Math::Vector3f * c, Material * material, const Math::Vector3f * normals = nullptr)
			: m_material(material)
		{
			m_vertex[0] = a ;
			m_vertex[1] = b ;
			m_vertex[2] = c ;
			std::fill(m_textureCoordinates, m_textureCoordinates + 3, nullptr);
			update() ;
			if (normals != nullptr)
			{
				::std::copy(normals, normals + 3, m_vertexNormal);
			}
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Triangle::Triangle()
		///
		/// \brief	Default constructor.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ Triangle()
		{
			std::fill(m_vertex, m_vertex + 3, nullptr);
			std::fill(m_textureCoordinates, m_textureCoordinates + 3, nullptr);
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Material * Triangle::material() const
		///
		/// \brief	Gets the material.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \return	null if it fails, else.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Material * material() const
		{ return m_material ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector3 const & Triangle::vertex(int i) const
		///
		/// \brief	Gets the ith vertex
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	i	Vertex index in [0;2].
		///
		/// \return	the vertex.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Math::Vector3f const & vertex(int i) const
		{ 
			assert(i >= 0 && i < 3);
			return *(m_vertex[i]) ; 
		}

		/// <summary>
		/// Returns the center of this triangle.
		/// </summary>
		/// <returns>The center of the triangle.</returns>
		__device__ __host__ Math::Vector3f center() const
		{
			return (vertex(0) + vertex(1) + vertex(2)) / 3;
		}

		/// <summary>
		/// Gets the textures coordinates of a vertex.
		/// </summary>
		/// <param name="i">The i.</param>
		/// <returns></returns>
		__device__ __host__ Math::Vector2f const & textureCoordinate(int i) const
		{
			assert(i >= 0 && i < 3);
			return *(m_textureCoordinates[i]);
		}

		/// <summary>
		/// Interpolates the texture coordinate given the u,v coordinates of an intersection.
		/// </summary>
		/// <param name="u">The u.</param>
		/// <param name="v">The v.</param>
		/// <returns></returns>
		__device__ __host__ Math::Vector2f interpolateTextureCoordinate(float u, float v) const
		{
			return textureCoordinate(0)*(1 - u - v) + textureCoordinate(1)*u + textureCoordinate(2)*v;
		}

		/// <summary>
		/// Samples the texture given the u,v coordinates of an intersection.
		/// </summary>
		/// <param name="u">The u.</param>
		/// <param name="v">The v.</param>
		/// <returns>The color of the texture at the given u,v coordinates or (1.0,1.0,1.0) if no texture is associated with the material.</returns>
		__device__ __host__ RGBColor sampleTexture(float u, float v) const
		{
			if(m_material->hasTexture() && m_textureCoordinates[0]!=NULL)
			{
				RGBColor texel = m_material->getTexture().pixel(interpolateTextureCoordinate(u, v));
				return texel;
			}
			return RGBColor(1.0f, 1.0f, 1.0f);
		}

		/// <summary>
		/// Samples the triangle given the u,v coordinates of an intersection
		/// </summary>
		/// <param name="u"></param>
		/// <param name="v"></param>
		/// <returns></returns>
		__device__ __host__ Math::Vector3f samplePoint(float u, float v) const
		{
			return m_uAxis*u + m_vAxis*v + m_vertex0;
		}

		/// <summary>
		/// Samples the normal given the u,v coordinates of an intersection. The normal is oriented toward the 'toward' point.
		/// </summary>
		/// <param name="u">The u coordinate.</param>
		/// <param name="v">The v coordinate.</param>
		/// <param name="toward">The toward point.</param>
		/// <returns></returns>
		__device__ __host__ Math::Vector3f sampleNormal(float u, float v, const Math::Vector3f & toward) const
		{
			Math::Vector3f result = (m_vertexNormal[0]*(1 - u - v) + m_vertexNormal[1]*u + m_vertexNormal[2]*v).normalized();
			if ((toward - (m_vertex0+m_uAxis*u+m_vAxis*v))*result < 0.0)
			{
				return (result*(-1.0)).normalized();
			}
			return result.normalized();
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	const Math::Vector3 & Triangle::uAxis() const
		///
		/// \brief	Gets the u axis.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \return	.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ const Math::Vector3f & uAxis() const
		{ 
			return m_uAxis ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	const Math::Vector3 & Triangle::vAxis() const
		///
		/// \brief	Gets the v axis.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \return	.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ const Math::Vector3f & vAxis() const
		{ 
			return m_vAxis ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	const Math::Vector3 & Triangle::normal() const
		///
		/// \brief	Gets the normal.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \return	.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ const Math::Vector3f & normal() const
		{ return m_normal ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector3 Triangle::normal(Math::Vector3 const & point) const
		///
		/// \brief	Gets the normal directed toward the half space containing the provided point.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	point	The point.
		///
		/// \return	.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Math::Vector3f normal(Math::Vector3f const & point) const
		{
			if((point-m_vertex0)*m_normal<0.0)
			{ return m_normal*(-1.0) ; }
			return m_normal ; 
		}

		/// <summary>
		/// Returns the direction of a reflected ray, from a surface normal and the direction of the incident ray.
		/// </summary>
		/// <param name="n">The n.</param>
		/// <param name="dir">The dir.</param>
		/// <returns></returns>
		__device__ __host__ static Math::Vector3f reflectionDirection(Math::Vector3f const & n, Math::Vector3f const & dir)
		{
			Math::Vector3f reflected(dir - n*(2.0f*(dir*n)));
			return reflected;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector3 Triangle::reflectionDirection(Math::Vector3 const & dir) const
		///
		/// \brief Returns the direction of a reflected ray, from the direction of the incident ray.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	dir	The direction of the incident ray.
		///
		/// \return	The direction of the reflected ray.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Math::Vector3f reflectionDirection(Math::Vector3f const & dir) const
		{
			Math::Vector3f n = normal();
			Math::Vector3f reflected(dir-n*(2.0f*(dir*n))) ; 
			return reflected ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector3 Triangle::reflectionDirection(Ray const & ray) const
		///
		/// \brief	Returns the direction of the reflected ray from a ray description.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	ray	The incident ray.
		///
		/// \return	.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Math::Vector3f reflectionDirection(Ray const & ray) const
		{
			Math::Vector3f n = normal() ;
			//if(n*(ray.source()-vertex(0))<=0.0)
			if(n*(ray.source()-m_vertex0)<=0.0)
			{ n = n*(-1.0) ; }
			Math::Vector3f reflected(ray.direction()-n*(2.0f*(ray.direction()*n))) ; 
			return reflected ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	bool Triangle::intersection(Ray const & r, float & t, float & u, float & v) const
		///
		/// \brief	Computes the intersection between a ray and this triangle.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	r		 	The tested ray.
		/// \param 	t	The distance between the ray source and the intersection point (t>=0).
		/// \param	u		 	The u coordinate of the intersection (useful for texture mapping).
		/// \param	v		 	The v coordinate of the intersection (useful for texture mapping).
		///
		/// \return	True if an intersection has been found, false otherwise.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ bool intersection(Ray const & r, float & t, float & u, float & v) const
		{
			/* find vectors for two edges sharing vert0 */
			const Math::Vector3f & edge1(uAxis()) ;
			const Math::Vector3f & edge2(vAxis()) ;

			/* begin calculating determinant - also used to calculate U parameter */
			Math::Vector3f pvec(r.direction() ^ edge2);

			/* if determinant is near zero, ray lies in plane of triangle */
			float det = edge1 * pvec ;
		
			if (fabs(det)<0.000000001) // if(det > -0.000001 && det < 0.000001) 
			{
				return false ; 
			}

			float inv_det = 1.0f / det;

			/* calculate distance from vert0 to ray origin */
			//Math::Vector3 tvec(r.source() - vertex(0));
			Math::Vector3f tvec(r.source() - m_vertex0);

			/* calculate U parameter and test bounds */
			u = (tvec * pvec) * inv_det;

			//std::cout<<"u = "<<u<<std::endl ;

			if (fabs(u-0.5)>0.5) //u < 0.0 || u > 1.0) //
			{
				return  false ;
			}

			/* prepare to test V parameter */
			Math::Vector3f qvec(tvec ^ edge1) ;

			/* calculate V parameter and test bounds */
			v = (r.direction() * qvec) * inv_det;
			if (v < 0.0 || u + v > 1.0)
			{
				return false ;
			}

			/* calculate t, ray intersects triangle */
			t = (edge2 * qvec) * inv_det;

			return t>=0.0001 ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	bool Triangle::generalIntersection(Ray const & r, float & t, float & u, float & v) const
		///
		/// \brief	Computes the intersection between the ray and the plane supporting the triangle.
		///
		/// \author	F. Lamarche, Universit� de Rennes 1
		/// \date	04/12/2013
		///
		/// \param	r	The ray.
		/// \param	t	The distance between the ray source and the intersection.
		/// \param	u	The u coordinate of the intersection.
		/// \param	v	The v coordinate of the intersection.
		///
		/// \return	True if the ray is not parallel to the plane.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ bool generalIntersection(Ray const & r, float & t, float & u, float & v) const
		{
			/* find vectors for two edges sharing vert0 */
			const Math::Vector3f & edge1(uAxis()) ;
			const Math::Vector3f & edge2(vAxis()) ;
			float det,inv_det;

			/* begin calculating determinant - also used to calculate U parameter */
			Math::Vector3f pvec(r.direction() ^ edge2);

			/* if determinant is near zero, ray lies in plane of triangle */
			det = edge1 * pvec ;
		
			if (det > -0.000001 && det < 0.000001) 
			{
				return false ; 
			}

			inv_det = 1.0f / det;

			/* calculate distance from vert0 to ray origin */
			Math::Vector3f tvec(r.source() - m_vertex0);

			/* calculate U parameter and test bounds */
			u = (tvec * pvec) * inv_det;

			/* prepare to test V parameter */
			Math::Vector3f qvec(tvec ^ edge1) ;

			/* calculate V parameter and test bounds */
			v = (r.direction() * qvec) * inv_det;

			/* calculate t, ray intersects triangle */
			t = (edge2 * qvec) * inv_det;

			return true ;
		}

		/// <summary>
		///  Returns the surface of the triangle
		/// </summary>
		/// <returns></returns>
		__device__ __host__ float surface() const
		{
			return (m_uAxis^m_vAxis).norm() / 2.0f;
		}

		/// <summary>
		/// Computes random barycentric coordinates
		/// </summary>
		/// <returns></returns>
		__device__ __host__ Math::Vector3f randomBarycentric() const
		{
			float r = (float)rand() / RAND_MAX;
			float s = (float)rand() / RAND_MAX;
			float a = float(1.0) - sqrt(s);
			float b = (float)((1.0 - r)*sqrt(s));
			float c = r*sqrt(s);
			return Math::makeVector(a, b, c);
		}

		/// <summary>
		/// Computes a point on a triangle from the barycentric coordinates
		/// </summary>
		/// <param name="barycentric"> The barycentric coordinates of the point</param>
		/// <returns></returns>
		__device__ __host__ Math::Vector3f pointFromBraycentric(const Math::Vector3f & barycentric) const
		{
			Math::Vector3f tmp = randomBarycentric();
			return ((*m_vertex[0]) * tmp[0] + (*m_vertex[1]) * tmp[1] + (*m_vertex[2]) * tmp[2]);
		}

		/// <summary>
		/// Samples the texture from the provided barycentric coordinates
		/// </summary>
		/// <param name="barycentic"> The barycentric coordinates of the point</param>
		/// <returns> The color of the texture at the given barycentric coordinates</returns>
		__device__ __host__ RGBColor sampleTexture(const Math::Vector3f & barycentic) const
		{
			if (m_material->hasTexture())
			{
				Math::Vector2f textureCoord = textureCoordinate(0)*barycentic[0] + textureCoordinate(1)*barycentic[1] + textureCoordinate(2)*barycentic[2];
				return m_material->getTexture().pixel(textureCoord);
			}
			return RGBColor(1.0, 1.0, 1.0);
		}


		/// <summary>
		///  Computes a random point on the triangle
		/// </summary>
		/// <returns></returns>
		__device__ __host__ Math::Vector3f randomPoint() const
		{
			float r = (float)rand() / RAND_MAX; 
			float s = (float)rand() / RAND_MAX; 
			float a = float(1.0) - sqrt(s);
			float b = (float)((1.0 - r)*sqrt(s));
			float c = r*sqrt(s);
			return ((*m_vertex[0]) * a + (*m_vertex[1]) * b + (*m_vertex[2]) * c);
		}

		/// <summary>
		/// Computes the distance between the point and the plane on which the triangle lies.
		/// </summary>
		/// <param name="point"></param>
		/// <returns></returns>
		__device__ __host__ float planeDistance(const Math::Vector3f & point) const
		{
			return fabs((point - m_vertex0)*m_normal);
		}
	} ;

#endif
 
