namespace Tools {

	inline __device__ __host__ unsigned int coords_to_index(unsigned int i, unsigned int j, unsigned int w, unsigned int h)
	{
		return i * w + j;
	}
}