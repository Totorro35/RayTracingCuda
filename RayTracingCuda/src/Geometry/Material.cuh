#ifndef _Geometry_Material_H
#define _Geometry_Material_H

#include <Geometry/RGBColor.cuh>
#include <Geometry/Texture.cuh>
#include <Math/Vectorf.cuh>

	/** \brief A material definition. */
	class Material
	{
	protected:
		/// <summary> The ambient color </summary>
		RGBColor m_ambientColor ;
		/// <summary> the diffuse color</summary>
		RGBColor m_diffuseColor ;
		/// <summary> The specular color</summary>
		RGBColor m_specularColor ;
		/// <summary> The shininess</summary>
		double    m_shininess ;
		/// <summary> The emissive color</summary>
		RGBColor m_emissiveColor ;
		/// <summary> The filename of the itexture image</summary>
		std::string m_textureFile;
		/// <summary> The texture or nullptr if no texture is associated with this material</summary>
		Texture * m_texture;

		//Coefficient d'absorption;
		double m_alpha;

		double m_fresnel=1/2;

		int m_id=0;

		double m_taille = 0.5;
		double m_repartition = 0.5;

	public:

		int getId() {
			return m_id;
		}

		void setId(int id) {
			m_id = id;
		}

		double getTaille() {
			return m_taille;
		}

		void setTaille(double taille) {
			m_taille = taille;
		}

		double getFresnel() {
			return m_fresnel;
		}

		void setFresnel(double fresnel) {
			m_fresnel = fresnel;
		}

		double getRepartition() {
			return m_repartition;
		}

		void setRepartition(double repartition) {
			m_repartition = repartition;
		}

		__device__ __host__ Material(RGBColor const & ambientColor = RGBColor(), RGBColor const & diffuseColor = RGBColor(),
				 RGBColor specularColor = RGBColor(), double shininess = 1.0, RGBColor const & emissiveColor = RGBColor(), std::string const & textureFile = "", double alpha=0.1)
				 : m_ambientColor(ambientColor), m_diffuseColor(diffuseColor), m_specularColor(specularColor),
				   m_shininess(shininess), m_emissiveColor(emissiveColor), m_textureFile(textureFile), m_texture(NULL), m_alpha(alpha)
		{}

		/// <summary>
		/// Sets the ambient color.
		/// </summary>
		/// <param name="color">The color.</param>
		__device__ __host__ void setAmbient(RGBColor const & color)
		{
			m_ambientColor = color;
		}

		/// <summary>
		/// Gets the ambient color.
		/// </summary>
		/// <returns>The ambiant color</returns>
		__device__ __host__ const RGBColor & getAmbient() const
		{ return m_ambientColor ; }

		/// <summary>
		/// Sets the diffuse color.
		/// </summary>
		/// <param name="color">The color.</param>
		__device__ __host__ void setDiffuse(RGBColor const & color)
		{
			m_diffuseColor = color;
		}
		
		/// <summary>
		/// Gets the diffuse color.
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const RGBColor & getDiffuse() const
		{ return m_diffuseColor ; }

		/// <summary>
		/// Sets the specular color.
		/// </summary>
		/// <param name="color">The color.</param>
		__device__ __host__ void setSpecular(RGBColor const & color)
		{
			m_specularColor = color;
		}

		/// <summary>
		/// Gets the specular color.
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const RGBColor & getSpecular() const
		{ return m_specularColor ; }

		/// <summary>
		/// Sets the shininess.
		/// </summary>
		/// <param name="s">The shininess.</param>
		__device__ __host__ void setShininess(double s)
		{
			m_shininess = s;
		}

		/// <summary>
		/// Gets the shininess.
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const double & getShininess() const
		{ return m_shininess ; }

		/// <summary>
		/// Sets the emissive color.
		/// </summary>
		/// <param name="color">The color.</param>
		__device__ __host__ void setEmissive(RGBColor const & color)
		{
			m_emissiveColor = color;
		}

		/// <summary>
		/// Gets the emissive color.
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const RGBColor & getEmissive() const
		{ return m_emissiveColor ; }

		/// <summary>
		/// Sets the texture file.
		/// </summary>
		/// <param name="textureFile">The texture file.</param>
		__host__ void setTextureFile(const ::std::string & textureFile)
		{
			m_textureFile = textureFile;
			::std::cout << "Loading texture: "<< m_textureFile << "..." << ::std::flush;
			m_texture = new Texture(m_textureFile);
			if (!m_texture->isValid())
			{
				delete m_texture;
				m_texture = NULL;
				::std::cout << "discarded" << ::std::endl;
			}
			else
			{
				::std::cout << "OK" << ::std::endl;
			}
		}

		/// <summary>
		/// Gets the texture file.
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const ::std::string & getTextureFile() const
		{
			return m_textureFile;
		}

		/// <summary>
		/// Returns the texture
		/// </summary>
		/// <returns></returns>
		__device__ __host__ const Texture & getTexture() const
		{
			return *m_texture;
		}

		/// <summary>
		/// Tests if a texture is associated with this material.
		/// </summary>
		/// <returns> true if this materail has a texture, false otherwise</returns>
		__device__ __host__ bool hasTexture() const
		{
			return m_texture != NULL;
		}

		__device__ __host__ double getAbsorption() {
			return m_alpha;
		}

		__device__ __host__ void setAbsorption(double alpha) {
			m_alpha = alpha;
		}
		/*
		RGBColor BRDF(RGBColor textureColor, Math::Vector3f entree, Math::Vector3f sortie, Math::Vector3f normal) {
			RGBColor result = this->getDiffuse()*textureColor;
			float factor = 1.f;
			//float factor = (pow(abs(reflectionDirection(normal, entree)*sortie), 3) - pow(0.9*abs(reflectionDirection(normal, entree)*sortie), 5))/0.37;
			return result*factor;
		}

		static Math::Vector3f reflectionDirection(Math::Vector3f const & n, Math::Vector3f const & dir)
		{
			Math::Vector3f reflected(dir - n * (2.0f*(dir*n)));
			return reflected;
		}*/

	};

#endif
