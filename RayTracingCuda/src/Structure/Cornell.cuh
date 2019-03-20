
#include <vector>
#include "Geometry/Triangle.cuh"
#include "Geometry/Material.cuh"
#include "Geometry/Geometry.cuh"

class Square : public Geometry {
public:
	Square(Material * material)
		: Geometry()
	{
		int p0 = addVertex(Math::makeVector(0.5, 0.5, 0.0));
		int p1 = addVertex(Math::makeVector(0.5, -0.5, 0.0));
		int p2 = addVertex(Math::makeVector(-0.5, 0.5, 0.0));
		int p3 = addVertex(Math::makeVector(-0.5, -0.5, 0.0));
		addTriangle(p0, p1, p2, material);
		addTriangle(p1, p2, p3, material);
	}
};

class Disk : public Geometry {
public:
	Disk(int nbDiv, Material * material)
	{
		unsigned int center = addVertex(Math::Vector3f());
		::std::vector<unsigned int> vertices;
		for (int cpt = 0; cpt < nbDiv; cpt++)
		{
			float angle = float((2.0f*M_PI / nbDiv)*cpt);
			int i = addVertex(Math::makeVector(cos(angle), sin(angle), 0.0f));
			vertices.push_back(i);
		}
		for (int cpt = 0; cpt < nbDiv; cpt++)
		{
			addTriangle(center, vertices[cpt], vertices[(cpt + 1) % nbDiv], material);
		}
	}
};

class Cylinder : public Geometry {
public :
	Cylinder(int nbDiv, float scaleDown, float scaleUp, Material * material)
	{
		Disk disk1(nbDiv, material);
		disk1.scale(scaleUp);
		disk1.translate(Math::makeVector(0.0f, 0.0f, 0.5f));

		Disk disk2(nbDiv, material);
		disk2.scale(scaleDown);
		disk2.translate(Math::makeVector(0.0f, 0.0f, -0.5f));

		merge(disk1);
		merge(disk2);


		for (int cpt = 0; cpt < nbDiv; cpt++)
		{
			addTriangle(disk1.getVertices()[cpt], disk1.getVertices()[(cpt + 1) % nbDiv], disk2.getVertices()[cpt], material);
			addTriangle(disk1.getVertices()[(cpt + 1) % nbDiv], disk2.getVertices()[cpt], disk2.getVertices()[(cpt + 1) % nbDiv], material);
		}
	}
};

class Cube : public Geometry {
public:
	Cube(Material * material)
		: Geometry()
	{
		Square sq0(material);
		sq0.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		merge(sq0);

		Square sq1(material);
		sq1.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq1.rotate(Math::Quaternion<float>(Math::makeVector(1.0f, 0.0f, 0.0f), (float)M_PI / 2.0f));
		merge(sq1);

		Square sq2(material);
		sq2.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq2.rotate(Math::Quaternion<float>(Math::makeVector(1.0f, 0.0f, 0.0f), (float)M_PI));
		merge(sq2);

		Square sq3(material);
		sq3.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq3.rotate(Math::Quaternion<float>(Math::makeVector(1.0f, 0.0f, 0.0f), (float)-M_PI / 2.0f));
		merge(sq3);

		Square sq4(material);
		sq4.translate(Math::makeVector(0.0, 0.0, 0.5));
		sq4.rotate(Math::Quaternion<float>(Math::makeVector(0.0f, 1.0f, 0.0f), (float)M_PI / 2.0f));
		merge(sq4);

		Square sq5(material);
		sq5.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq5.rotate(Math::Quaternion<float>(Math::makeVector(0.0f, 1.0f, 0.0f), (float)-M_PI / 2.0f));
		merge(sq5);

	}
};

class Cornell : public Geometry {

public:

	Cornell(Material * up, Material * down, Material * front, Material * back, Material * left, Material * right)
		: Geometry()
	{
		Square sq0(up);
		sq0.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		merge(sq0);

		Square sq1(left);
		sq1.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq1.rotate(Math::Quaternion<float>(Math::makeVector(1.0f, 0.0f, 0.0f), (float)M_PI / 2.0f));
		merge(sq1);

		Square sq2(down);
		sq2.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq2.rotate(Math::Quaternion<float>(Math::makeVector(1.0f, 0.0f, 0.0f), (float)M_PI));
		merge(sq2);

		Square sq3(right);
		sq3.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq3.rotate(Math::Quaternion<float>(Math::makeVector(1.0f, 0.0f, 0.0f), (float)-M_PI / 2.0f));
		merge(sq3);

		Square sq4(front);
		sq4.translate(Math::makeVector(0.0, 0.0, 0.5));
		sq4.rotate(Math::Quaternion<float>(Math::makeVector(0.0f, 1.0f, 0.0f), (float)M_PI / 2.0f));
		merge(sq4);

		Square sq5(back);
		sq5.translate(Math::makeVector(0.0f, 0.0f, 0.5f));
		sq5.rotate(Math::Quaternion<float>(Math::makeVector(0.0f, 1.0f, 0.0f), (float)-M_PI / 2.0f));
		merge(sq5);
	}

	static std::vector<Triangle> generateCornell() {
		std::vector<Triangle> result;

		Material ocre(RGBColor(), RGBColor(1.0, 0.87, 0.53), RGBColor(0, 0, 0), 4, RGBColor(), "", 0.4f);
		ocre.setId(0);
		Material* d_ocre;
		cudaMalloc((void**)&d_ocre, sizeof(Material));
		cudaMemcpy(d_ocre, &ocre, sizeof(Material), cudaMemcpyHostToDevice);

		Material pourpre(RGBColor(), RGBColor(0.70, 0.13, 0.13), RGBColor(0, 0, 0), 4, RGBColor(), "", 0.4f);
		pourpre.setId(0);
		Material* d_pourpre;
		cudaMalloc((void**)&d_pourpre, sizeof(Material));
		cudaMemcpy(d_pourpre, &pourpre, sizeof(Material), cudaMemcpyHostToDevice);

		Material  emeraude(RGBColor(), RGBColor(0.07, 0.72, 0.29), RGBColor(0, 0, 0), 4, RGBColor(), "", 0.4f);
		emeraude.setId(0);
		Material* d_emeraude;
		cudaMalloc((void**)&d_emeraude, sizeof(Material));
		cudaMemcpy(d_emeraude, &emeraude, sizeof(Material), cudaMemcpyHostToDevice);

		Material  ivoire(RGBColor(), RGBColor(1.0, 1.0, 1.0), RGBColor(0, 0, 0), 4, RGBColor(1.0, 1.0, 1.0), "", 0.4f);
		Material* d_ivoire;
		cudaMalloc((void**)&d_ivoire, sizeof(Material));
		cudaMemcpy(d_ivoire, &ivoire, sizeof(Material), cudaMemcpyHostToDevice);

		Material  turquoise(RGBColor(), RGBColor(0.06, 157 / 255., 232 / 255.), RGBColor(0, 0, 0), 100, RGBColor(), "", 0.4f);
		turquoise.setId(0);
		Material* d_turquoise;
		cudaMalloc((void**)&d_turquoise, sizeof(Material));
		cudaMemcpy(d_turquoise, &turquoise, sizeof(Material), cudaMemcpyHostToDevice);

		Material  ebene(RGBColor(), RGBColor(53 / 255., 53 / 255., 52 / 255.), RGBColor(0, 0, 0), 4, RGBColor(), "", 0.4f);
		Material* d_ebene;
		cudaMalloc((void**)&d_ebene, sizeof(Material));
		cudaMemcpy(d_ebene, &ebene, sizeof(Material), cudaMemcpyHostToDevice);

		Material  miroir_material(RGBColor(), RGBColor(1.0, 1.0, 1.0), RGBColor(0.0, 0.0, 0.0), 100000, RGBColor(), "", 0.4f);
		Material* d_miroir_material;
		cudaMalloc((void**)&d_miroir_material, sizeof(Material));
		cudaMemcpy(d_miroir_material, &miroir_material, sizeof(Material), cudaMemcpyHostToDevice);


		Cornell geo(d_ocre, d_ocre, d_ocre, d_ocre, d_emeraude, d_pourpre);
		geo.scaleX(10);
		geo.scaleY(10);
		geo.scaleZ(10);
		for (const Triangle & tri : geo.getTriangles()) {
			result.push_back(tri);
		}

		Square light(d_ivoire);
		light.translate(Math::makeVector(0.0, 0.0, 4.99));
		light.scaleX(3);
		light.scaleY(3);
		for (const Triangle & tri : light.getTriangles()) {
			result.push_back(tri);
		}

		Cylinder cylinder(1000, 0.7, 0.7, d_turquoise);
		cylinder.scaleZ(5);
		cylinder.translate(Math::makeVector(3.0, 2.0, -2.5));
		/*for (const Triangle & tri : cylinder.getTriangles()) {
			result.push_back(tri);
		}*/

		Square miroir(d_miroir_material);
		Math::Quaternion<float> r(Math::makeVector(0.0f, 1.0f, 0.0f), 67.5);
		miroir.rotate(r);
		miroir.translate(Math::makeVector(4.99f, 0.0f, -0.1f));
		miroir.scaleZ(8);
		miroir.scaleY(8);
		for (const Triangle & tri : miroir.getTriangles()) {
			result.push_back(tri);
		}

		Cube table(d_ebene);
		table.scaleZ(0.1);
		table.scaleY(4);
		table.translate(Math::makeVector(1.0, -3.0, -2.));
		for (const Triangle & tri : table.getTriangles()) {
			result.push_back(tri);
		}
		return result;
	}
};