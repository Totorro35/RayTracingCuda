#pragma once

#include <SDL.h>
#include "cuda_runtime.h"
#include <iostream>
#include "Geometry/RGBColor.cuh"

namespace kernel
{
	__global__ void convert_image(uint8_t * d_ucfb, const RGBColor * d_fb, size_t width, size_t height,int nb_pass)
	{
		const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
		const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
		const unsigned int index = i * width + j;
		const unsigned int uc_index = 4 * index;
		if (i < height && j < width)
		{
			RGBColor pixel = d_fb[index]/float(nb_pass);
			pixel = pixel.toneMappe();
			uint8_t r = pixel[0];
			uint8_t g = pixel[1];
			uint8_t b = pixel[2];
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
			d_ucfb[uc_index + 0] = b;
			d_ucfb[uc_index + 1] = g;
			d_ucfb[uc_index + 2] = r;
#else
			//TODO
			d_ucfb[uc_index + 0] = r;
			d_ucfb[uc_index + 1] = g;
			d_ucfb[uc_index + 2] = b;
			d_ucfb[uc_index + 3] = 255;
#endif
		}
	}

	__global__ void print(int* a) {

	}
}

class Visualizer
{
protected:

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
#define rmask = 0xff000000;
#define gmask = 0x00ff0000;
#define bmask = 0x0000ff00;
#define amask = 0x000000ff;
#else
#define rmask = 0x000000ff;
#define gmask = 0x0000ff00;
#define bmask = 0x00ff0000;
#define amask = 0xff000000;
#endif

	size_t m_window_width;
	size_t m_window_height;

	SDL_Window * m_window;

	SDL_Surface * m_window_surface;

	uint8_t * d_ucfb;

public:


	Visualizer(size_t width, size_t height) :
		m_window_width(width),
		m_window_height(height)
	{
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
		{
			::std::cerr << "Critical error" << ::std::endl;
			::std::cerr << "SDL_Init problem: " << SDL_GetError() << ::std::endl;
			exit(1);
		}
		atexit(SDL_Quit);

		m_window = SDL_CreateWindow("GPURT", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, int(m_window_width), int(m_window_height), 0);

		m_window_surface = SDL_GetWindowSurface(m_window);
		if (m_window_surface == NULL)
		{
			std::cerr << "Critical Error!" << std::endl;
			std::cerr << "Could not get the frame buffer!" << std::endl;
			std::cerr << SDL_GetError() << std::endl;
			exit(1);
		}

		cudaError_t error = cudaMalloc((void**)&d_ucfb, m_window_height * m_window_width * 4 * sizeof(uint8_t));
		if (error != cudaSuccess)
		{
			std::cerr << "Critical error, could not create the frame buffer" << std::endl;
			std::cerr << error << std::endl;
			exit(1);
		}
	}


	~Visualizer()
	{
		SDL_DestroyWindow(m_window);
		cudaFree(d_ucfb);
	}

	void blit(const RGBColor * d_fb, dim3 const & block_size, dim3 const& grid_size, int nb_pass)
	{
		kernel::convert_image << <grid_size, block_size >> > (d_ucfb, d_fb, m_window_width, m_window_height, nb_pass);
		cudaError_t error = cudaMemcpy(m_window_surface->pixels, d_ucfb, 4 * m_window_height * m_window_width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			std::cerr << "Error, could not blit the frame buffer!" << std::endl;
			std::cerr << error << std::endl;
		}
	}

	inline float proportion() {
		return float(m_window_height) / float(m_window_width);
	}

	void waitKeyPressed()const
	{

		SDL_Event event;
		bool done = false;
		while (!done) {
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_KEYDOWN:
					/*break;*/
				case SDL_QUIT:
					done = true;
					break;
				default:
					break;
				}
			}/*while*/
		}/*while(!done)*/
	}


	void update()
	{
		SDL_UpdateWindowSurface(m_window);
	}

	inline size_t getWidth() const{
		return m_window_width;
	}

	inline size_t getHeight() const{
		return m_window_height;
	}
};

