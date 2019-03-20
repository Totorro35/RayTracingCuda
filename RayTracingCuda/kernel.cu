
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Visualizer/Visualizer.cuh"
#include "Geometry/Scene.cuh"

#include <stdio.h>
#include <chrono>

inline __device__ __host__ unsigned int divide_up(unsigned int num, unsigned int denum);

void update(bool * keys);

bool updateCamera(bool * keys,Camera& cam,float dt);

int main(int argc, char *argv[])
{
	const unsigned int k = 1;
	const unsigned int width = 1080 * k;
	const unsigned int height = 720 * k;
	const unsigned int num_pixel = width * height;

	Visualizer visu(width, height);

	Scene scene(&visu);

	const dim3 block_size(4, 8);
	const dim3 grid_size = dim3(divide_up(height, block_size.x), divide_up(width, block_size.y));

	RGBColor * d_fbf;
	cudaMalloc((void**)&d_fbf, num_pixel * sizeof(RGBColor));

	Camera cam(Math::makeVector(-4.0f, 0.0f, 0.0f), Math::makeVector(-3.0f, 0.0f, 0.0f), 0.4f, 1.0f, 9.f / 16.f);
	scene.setCam(cam);

	bool keys[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	int nb_pass = 1;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	while (1) {

		t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);

		float dt = time_span.count();
		t1 = t2;

		update(keys);

		bool reset = updateCamera(keys,cam,dt);

		if (reset) {
			scene.setCam(cam);
			nb_pass = 1;
		}

		scene.compute(d_fbf, grid_size, block_size, nb_pass,reset);
		cudaDeviceSynchronize();

		visu.blit(d_fbf, block_size, grid_size,nb_pass);
		visu.update();

		nb_pass++;
		std::cout << 1.f / dt << std::endl;
	}

	visu.waitKeyPressed();

	//Liberation GPU
	cudaFree(d_fbf);

	cudaDeviceReset();

	//Liberation CPU

    return 0;
}

inline __device__ __host__ unsigned int divide_up(unsigned int num, unsigned int denum)
{
	unsigned int res = num / denum;
	if (res * denum < num)
	{
		res += 1;
	}
	return res;
}

void update(bool * keys)
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		if (event.type == SDL_KEYDOWN)
		{
			switch (event.key.keysym.sym)
			{
			case SDLK_UP:
				keys[0] = 1;
				break;
			case SDLK_DOWN:
				keys[1] = 1;
				break;
			case SDLK_LEFT:
				keys[2] = 1;
				break;
			case SDLK_RIGHT:
				keys[3] = 1;
				break;
			case SDLK_z:
				keys[4] = 1;
				break;
			case SDLK_s:
				keys[5] = 1;
				break;
			case SDLK_d:
				keys[6] = 1;
				break;
			case SDLK_q:
				keys[7] = 1;
				break;
			case SDLK_SPACE:
				keys[8] = 1;
				break;
			case SDLK_LCTRL:
				keys[9] = 1;
				break;
			default:
				break;
			}
		}
		else if (event.type == SDL_KEYUP)
		{
			switch (event.key.keysym.sym)
			{
			case SDLK_UP:
				keys[0] = 0;
				break;
			case SDLK_DOWN:
				keys[1] = 0;
				break;
			case SDLK_LEFT:
				keys[2] = 0;
				break;
			case SDLK_RIGHT:
				keys[3] = 0;
				break;
			case SDLK_z:
				keys[4] = 0;
				break;
			case SDLK_s:
				keys[5] = 0;
				break;
			case SDLK_d:
				keys[6] = 0;
				break;
			case SDLK_q:
				keys[7] = 0;
				break;
			case SDLK_SPACE:
				keys[8] = 0;
				break;
			case SDLK_LCTRL:
				keys[9] = 0;
				break;
			default:
				break;
			}
		}
		else if (event.type == SDL_QUIT)
		{
			exit(0);
		}
	}
}

bool updateCamera(bool * keys,Camera& cam, float dt) {

	float forward = 0;
	float upward = 0;
	float rightward = 0;
	float inclination = acos(cam.getFront()[2]);
	float azimuth = atan2(cam.getFront()[1],cam.getFront()[0]);
	const float speed = 2;
	const float angle_speed = 2;

	if (keys[0])
	{
		inclination -= angle_speed * dt;
	}
	if (keys[1])
	{
		inclination += angle_speed * dt;
	}
	if (keys[2])
	{
		azimuth += angle_speed * dt;
	}
	if (keys[3])
	{
		azimuth -= angle_speed * dt;
	}
	if (keys[4])
	{
		forward += speed * dt;
	}
	if (keys[5])
	{
		forward -= speed * dt;
	}
	if (keys[6])
	{
		rightward += speed * dt;
	}
	if (keys[7])
	{
		rightward -= speed * dt;
	}
	if (keys[8])
	{
		upward += speed * dt;
	}
	if (keys[9])
	{
		upward -= speed * dt;
	}

	if (inclination > 3)
	{
		inclination = 3;
	}
	else if (inclination < 0.1)
	{
		inclination = 0.1;
	}

	Math::Vector3f translation = Math::makeVector(rightward, forward, upward);

	bool reset = translation != Math::makeVector(0.0f, 0.0f, 0.0f);
	reset = reset || inclination != 0 || azimuth != 0;

	if (reset) {
		cam.translateLocal(translation);
		Math::Vector3f direction = Math::makeVector(sin(inclination) * cos(azimuth), sin(inclination) * sin(azimuth), cos(inclination));
		cam.setTarget(cam.getPosition()+direction);

	}

	return reset;
}
