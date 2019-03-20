#pragma once

#include "Visualizer/Visualizer.cuh"
#include "device_launch_parameters.h"
#include "Geometry/RGBColor.cuh"
#include "cuda_runtime.h"
#include "Tools.cuh"
#include "Geometry/Ray.cuh"
#include "Geometry/Camera.cuh"
#include "Geometry/Material.cuh"
#include "Geometry/RayTriangleIntersection.cuh"
#include "Structure/Cornell.cuh"

#include <curand_kernel.h>
#include "Geometry/BRDFLib.cuh"

#include "Structure/BVH.cuh"

namespace scene {

	__global__ void render_init(int max_y, int max_x, curandState *rand_state) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= max_x) || (j >= max_y)) return;
		int pixel_index = j * max_x + i;
		//Each thread gets same seed, a different sequence number, no offset
		curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
	}

#define RANDVEC3 Math::makeVector(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

	__device__ Math::Vector3f random_in_unit_sphere(curandState *local_rand_state) {
		Math::Vector3f p;
		do {
			p = RANDVEC3*2.f - Math::makeVector(1.f, 1.f, 1.f);
		} while (p.norm2() >= 1.0f);
		return p;
	}

	__device__  float ombrage(const RayTriangleIntersection& intersection, PointLight const& light, const BVH* bvh) {

		RayTriangleIntersection mur;
		Math::Vector3f light_dir = (intersection.intersection() - light.position()).normalized();

		Ray ray(light.position(), light_dir);
		bvh->intersect(ray, mur);

		if (mur.valid()) {
			if (mur.triangle() == intersection.triangle()) {
				return 1.;
			}
		}

		return 0.;
	}

	__device__ __host__ PointLight samplingSphere(const PointLight& light0) {
		double phi = acos(sqrt(0.3));
		double theta = 2 * M_PI*0.3;

		Math::Vector3f translation = Math::getVector(theta, phi)*light0.getRadius();

		return PointLight(light0.position() + translation, light0.color());
	}

	__device__  RGBColor phongDirect(Ray const & ray, RayTriangleIntersection const & intersection, int depth,const BVH* bvh) {
		//Initialisation des variables
		RGBColor result(0.0, 0.0, 0.0); //Couleur de retour

		const Triangle* tri = intersection.triangle();
		Material* material = tri->material(); //Materiau du triangle
		Math::Vector3f normal = tri->sampleNormal(intersection.uTriangleValue(), intersection.vTriangleValue(), ray.source()); //Normal au triangle
		Math::Vector3f intersect_point = intersection.intersection(); //Point d'intersection sur le triangle
		Math::Vector3f viewer = (ray.source() - intersect_point).normalized(); //Direction de la source

		const PointLight* lights = bvh->getLights();

		for (int i = 0; i < bvh->getNumberLights();++i) {
			PointLight light = samplingSphere(lights[i]);

			//Valeur d'ombrage
			double shadow = ombrage(intersection, light,bvh);

			//Direction de la light
			double distance = (light.position() - intersect_point).norm();
			Math::Vector3f light_dir = (light.position() - intersect_point).normalized();
			if (normal*light_dir < 0)
			{
				continue;
			}

			double G = 1. / (distance + 1);

			RGBColor brdf = BRDFLib::computeColor(ray, intersection, light_dir);

			double proba_light = 1.;

			result = result + light.color() * brdf * G * shadow * proba_light;
		}
		delete[] lights;

		result = result / double(bvh->getNumberLights());

		//Ajout de la couleur d'Emission
		result = result + material->getEmissive() / (intersection.tRayValue() + 1);

		return result;
	}

	__device__  RGBColor sendRay(Ray const & ray, int depth, int maxDepth, const BVH* bvh, curandState *local_rand_state);

	__device__  RGBColor phongIndirect(Ray const & ray, RayTriangleIntersection const & intersection, int depth,int maxDepth, const BVH* bvh, curandState *local_rand_state) {
		RGBColor result;
		Material* material = intersection.triangle()->material();

		double rouletteRusse = 1.;
		double alpha = 0.;
		if (rouletteRusse > alpha) {
			Math::Vector3f normal = intersection.triangle()->sampleNormal(intersection.uTriangleValue(), intersection.vTriangleValue(), ray.source()).normalized();
			Math::Vector3f reflected = Triangle::reflectionDirection(normal, ray.direction()).normalized();
			Math::Vector3f indirect;
			do {
				indirect = random_in_unit_sphere(local_rand_state);
			} while (indirect*normal < 0);
			Ray rayIndirect(intersection.intersection() + indirect * 0.001, indirect);

			RGBColor brdf = BRDFLib::computeColor(ray, intersection, indirect);
			//RGBColor brdf(1.f,1.f,1.f);

			double pdf = 1.;

			RGBColor test = sendRay(rayIndirect, depth + 1, maxDepth, bvh, local_rand_state)*brdf / pdf;

			result = result + test;
		}

		result = result * (1 / (1 - alpha));

		return result;
	}

	

	__device__ RGBColor sendRay(Ray const & ray, int depth, int maxDepth,const BVH* bvh, curandState *local_rand_state)
	{
		RGBColor result(0.0, 0.0, 0.0);
		RGBColor brouillard(0.1f, 0.1f, 0.1f);
		float di = 0.;
		float df = 30.;
		bool brouill = false;

		if (depth <= maxDepth) {

			//Calcul de l'intersection
			RayTriangleIntersection intersection;
			if (bvh->intersect(ray, intersection)) {
				
				//Calcul des rayon reflechis
				Math::Vector3f normal = intersection.triangle()->sampleNormal(intersection.uTriangleValue(), intersection.vTriangleValue(), ray.source());

				Math::Vector3f reflected = Triangle::reflectionDirection(normal, ray.direction());
				Ray reflexion(intersection.intersection() + reflected * 0.001f, reflected);

				Material* material = intersection.triangle()->material();
				if (material == nullptr) {
					return RGBColor(intersection.tRayValue());
				}
				result = scene::phongDirect(ray, intersection, depth,bvh);

				result = result + scene::phongIndirect(ray, intersection, depth + 1, maxDepth, bvh, local_rand_state);

				if (brouill) {
					float d = intersection.tRayValue();
					float f = 0.0f;
					if (d < di) {
						f = 1.;
					}
					else if (d < df) {
						f = (df - d) / (df - di);
					}
					result = result * f + brouillard * (1 - f);
				}

			}

			else {
				if (brouill) {
					result = brouillard;
				}
				else {
					RGBColor background = RGBColor(1.0f,0.1f,0.1f);
					/*if (skybox->isValid()) {
						int u = (ray.direction().normalized()[1] + 1)*skybox->getSize()[0];
						int v = (ray.direction().normalized()[2] + 1)*skybox->getSize()[1];
						background = (skybox->pixel(u, v) / 10)*0.5 + brouillard * 0.8;
					}*/
					result = background;
				}

			}
		}
		return result;
	}

	__global__ void render(RGBColor* d_fb, const size_t height, const size_t width,const Camera* d_cam,const BVH* d_bvh, bool reset, curandState *rand_state)
	{
		const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
		const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i < height & j < width)
		{
			const unsigned index = Tools::coords_to_index(i, j, width, height);
			const float v = (float(i) + 0.5f) / float(height);
			const float u = (float(j) + 0.5f) / float(width);
			Ray ray = d_cam->getRay(u, v);

			RGBColor & pixel = d_fb[index];

			curandState* local_rand_state = &rand_state[index];

			if (reset) {
				pixel= sendRay(ray, 0, 5, d_bvh, local_rand_state);
			}
			else {
				pixel = pixel + sendRay(ray, 0, 5, d_bvh, local_rand_state);
			}
		}
	}
}

class Scene {

private :
	Visualizer* m_visu;

	Camera* d_cam;
	BVH* d_bvh;


public :

	Scene(Visualizer* visu) :
		m_visu(visu)
	{
		std::vector<Triangle> tri_scene=Cornell::generateCornell();

		BVH bvh(tri_scene);
		m_visu->waitKeyPressed();
		cudaMalloc((void**)&d_bvh, sizeof(BVH));
		cudaMemcpy(d_bvh, &bvh, sizeof(BVH), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_cam, sizeof(Camera));
	}

	~Scene(){}

	void setCam(const Camera& cam) {
		cudaMemcpy(d_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
	}

	void compute (RGBColor* d_fb,  dim3 const &  grid_size,  dim3 const &  block_size,int nb_pass,bool reset, curandState *rand_state)
	{
		//std::cout << "Pass "<<nb_pass<<std::endl;
		scene::render << < grid_size, block_size >> > (d_fb, m_visu->getHeight(), m_visu->getWidth(),d_cam,d_bvh, reset, rand_state);
	}

	static const int MAX_BOUNCES = 5;

};


