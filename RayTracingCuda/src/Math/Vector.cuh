#ifndef _Rennes1_Math_Vector_H
#define _Rennes1_Math_Vector_H

#include <iostream>
#include <cmath>
#include <algorithm>

namespace Math
{

	template <class Float, int dimensions>
	class Vector
	{
	protected:
		//! The scalars. 
		Float m_vector[dimensions] ;

	public:
		//! Iterator on vector coordinates. 
		typedef Float * iterator ;
		//! Const iterator on vector coordinates. 
		typedef const Float * const_iterator ;

		__device__ __host__ Vector(const Float * floatArray)
		{
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				m_vector[cpt] = floatArray[cpt] ;
			}
		}

		__device__ __host__ Vector(Float const & value = Float())
		{
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				m_vector[cpt] = value ;
			}
		}

		__device__ __host__ Vector(Vector<Float, dimensions> const & v)
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ m_vector[cpt] = v.m_vector[cpt] ; }
		}

		template <class Float2>
		__device__ __host__ Vector(Vector<Float2, dimensions> const & v)
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ m_vector[cpt] = Float(v[cpt]) ; }
		}

		__host__ __device__ __host__ Float * getBuffer() 
		{
			return m_vector;
		}

		__host__ __device__ __host__ const Float * getBuffer() const
		{
			return m_vector ;
		}

		inline __host__ __device__ __host__ Float norm() const
		{
			return (Float)sqrt(this->norm2()) ;
		}

		inline __host__ __device__ __host__ Float norm2() const
		{
			return (*this)*(*this) ;
		}

		__host__ __device__ __host__ const Float & operator[] (int index) const
		{ return m_vector[index] ; }

		__host__ __device__ __host__ Float & operator[] (int index)
		{ return m_vector[index] ; }

		__host__ __device__ __host__ Vector<Float, dimensions+1> push(Float const & coord) const
		{
			Math::Vector<Float, dimensions+1> result ;
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				result[cpt] = m_vector[cpt] ; 
			}
			result[dimensions] = coord ;
			return result ;
		}

		__device__ __host__ Vector<Float, dimensions> operator+ (Vector<Float, dimensions> const & v) const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ result[cpt] = m_vector[cpt]+v.m_vector[cpt] ;	}
			return result ;
		}

		__device__ __host__ Vector<Float, dimensions> operator- (Vector<Float, dimensions> const & v) const
		{
			Vector<Float,dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ result[cpt] = m_vector[cpt]-v.m_vector[cpt] ;	}
			return result ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> operator- () const
		///
		/// \brief	Negation operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	The result of the operation. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Vector<Float, dimensions> operator- () const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ result[cpt] = -m_vector[cpt] ;	}
			return result ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Float operator * (Vector<Float, dimensions> const & v) const
		///
		/// \brief	Scalar product. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	The result of the operation. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Float operator * (Vector<Float, dimensions> const & v) const
		{
			Float result ;
			result = m_vector[0]*v.m_vector[0] ;
			for(int cpt=1 ; cpt<dimensions ; cpt++)
			{ result = result + m_vector[cpt]*v.m_vector[cpt] ; }
			return result ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> operator* (Float const & v) const
		///
		/// \brief	Muliplication operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	The result of the operation. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Vector<Float, dimensions> operator* (Float const & v) const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ result[cpt] = m_vector[cpt]*v ;	}
			return result ;			
		}


		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> operator/ (Float const & v) const
		///
		/// \brief	Division operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	The result of the operation. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__device__ __host__ Vector<Float, dimensions> operator/ (Float const & v) const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ result[cpt] = m_vector[cpt]/v ;	}
			return result ;			
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator+= (Vector<Float, dimensions> const & v)
		///
		/// \brief	Assignment by addition operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> & operator+= (Vector<Float, dimensions> const & v)
		{
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				m_vector[cpt] += v.m_vector[cpt] ;
			}
			return (*this) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator-= (Vector<Float, dimensions> const & v)
		///
		/// \brief	Assignment by subtraction operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> & operator-= (Vector<Float, dimensions> const & v)
		{
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				m_vector[cpt] -= v.m_vector[cpt] ;
			}
			return (*this) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator*= (Float const & factor)
		///
		/// \brief	Assignment by muliplication operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	factor	the factor. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> & operator*= (Float const & factor)
		{
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				m_vector[cpt] *= factor ;
			}
			return (*this) ;				
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator/= (Float const & factor)
		///
		/// \brief	Assignment by division operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	factor	the factor. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> & operator/= (Float const & factor)
		{
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				m_vector[cpt] /= factor ;
			}
			return (*this) ;				
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator= (Vector<Float, dimensions> const & v)
		///
		/// \brief	Copy operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> & operator= (Vector<Float, dimensions> const & v)
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{
				m_vector[cpt]=v.m_vector[cpt] ;
			}
			return (*this) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator= (Vector<Float2, dimensions> const & v)
		///
		/// \brief	Generic copy operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <class Float2>
		__host__ __device__ __host__ Vector<Float, dimensions> & operator= (Vector<Float2, dimensions> const & v)
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{
				m_vector[cpt]=(Float)v[cpt] ;
			}
			return (*this) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> & operator= (Float const & s)
		///
		/// \brief	Copy operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	s	the. 
		///
		/// \return	A shallow copy of this object. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> & operator= (Float const & s)
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{ m_vector[cpt] = s ; }
			return *this ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> inv() const
		///
		/// \brief	Pseudo inverse of the vector. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> inv() const
		{ return (*this)/norm2() ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \property	bool operator<(Vector<Float, dimensions> const & v) const
		///
		/// \brief	 Comparison operator (lexicographical order). 
		///
		/// \return	The result of the comparison. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ bool operator<(Vector<Float, dimensions> const & v) const
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{
				if(m_vector[cpt]==v[cpt]) continue ;
				return m_vector[cpt]<v[cpt] ;
			}
			return false ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	bool operator==(Vector<Float, dimensions> const & v) const
		///
		/// \brief	Equality operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	true if the parameters are considered equivalent. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ bool operator==(Vector<Float, dimensions> const & v) const
		{
			for(int cpt=0 ; cpt<dimensions ; cpt++)
			{
				if(m_vector[cpt]!=v[cpt]) return false ;
			}
			return true ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	bool operator!=(Vector<Float, dimensions> const & v) const
		///
		/// \brief	Inequality operator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \param [in,out]	v	the v. 
		///
		/// \return	true if the parameters are not considered equivalent. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ bool operator!=(Vector<Float, dimensions> const & v) const
		{ return !((*this)==v) ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Vector<Float, dimensions> normalized() const
		///
		/// \brief	Return the associated normalized vector. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ Vector<Float, dimensions> normalized() const
		{
			return (*this)/norm() ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	int size() const
		///
		/// \brief	Returns the number of dimensions of this object. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		__host__ __device__ __host__ int size() const
		{ return dimensions ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	iterator begin()
		///
		/// \brief	Begin iterator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		iterator begin() 
		{ return m_vector ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	iterator end()
		///
		/// \brief End iterator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		iterator end()
		{ return m_vector+dimensions ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	const_iterator begin() const
		///
		/// \brief	Begin const iterator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		const_iterator begin() const 
		{ return &(m_vector[0]) ; }

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	const_iterator end() const
		///
		/// \brief	End const iterator. 
		///
		/// \author	Fabrice Lamarche, University Of Rennes 1
		/// \date	23/01/2010
		///
		/// \return	. 
		////////////////////////////////////////////////////////////////////////////////////////////////////
		const_iterator end() const
		{ return m_vector+dimensions ; }

		///////////////////////////////////////////////////////////////////////////////////
		/// \brief Computes absolute value of all vector components
		/// 
		/// \return 
		/// 
		/// \author F. Lamarche, University of Rennes 1.
		///////////////////////////////////////////////////////////////////////////////////
		Vector<Float, dimensions> simdAbs() const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				result.m_vector[cpt] = fabs(m_vector[cpt]) ;
			}
			return result ;
		}

		///////////////////////////////////////////////////////////////////////////////////
		/// \brief Computes the minimum value component by component
		/// 
		/// \param v
		/// \return 
		/// 
		/// \author F. Lamarche, University of Rennes 1.
		///////////////////////////////////////////////////////////////////////////////////
		Vector<Float, dimensions> simdMin(Vector<Float, dimensions> const & v) const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				result.m_vector[cpt] = ::std::min(m_vector[cpt], v.m_vector[cpt]) ;
			}
			return result ;
		}

		///////////////////////////////////////////////////////////////////////////////////
		/// \brief Computes the maximum value component by component
		/// 
		/// \param v
		/// \return 
		/// 
		/// \author F. Lamarche, University of Rennes 1.
		///////////////////////////////////////////////////////////////////////////////////
		Vector<Float, dimensions> simdMax(Vector<Float, dimensions> const & v) const
		{
			Vector<Float, dimensions> result ;
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				result.m_vector[cpt] = ::std::max(m_vector[cpt], v.m_vector[cpt]) ;
			}
			return result ;
		}

		Vector<Float, dimensions> zeroIfNegativeCoordinate() const
		{
			bool zero = false ;
			for(int cpt=0 ; cpt<dimensions ; ++cpt)
			{
				if(m_vector[cpt]<0.0) { return Vector<Float,dimensions>(Float(0.0)) ; }
			}
			return *this ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions+1> :::insert(int index, Float const & value) const
		///
		/// \brief	Creates a new vector by inserting a coordinate at position index.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \param	index	Zero-based index of the coordinate to insert.
		/// \param	value	The inserted value.
		///
		/// \return	The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions+1> insert(int index, Float const & value) const
		{
			Math::Vector<Float, dimensions+1> result ;
			for(int cpt=0 ; cpt<index ; ++cpt)
			{
				result[cpt] = (*this)[cpt] ;
			}
			result[index] = value ;
			for(int cpt=index ; cpt<dimensions ; ++cpt)
			{
				result[cpt+1] = (*this)[cpt] ;
			}
			return result ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions-1> :::remove(int index) const
		///
		/// \brief	Creates a new vector by removing the given coordinate.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \param	index	The index of the removed coordinate.
		///
		/// \return	The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions-1> remove(int index) const
		{
			Math::Vector<Float, dimensions-1> result ;
			for(int cpt=0 ; cpt<index ; ++cpt)
			{
				result[cpt] = (*this)[cpt] ;
			}
			for(int cpt=index+1 ; cpt<dimensions ; ++cpt)
			{
				result[cpt-1] = (*this)[cpt] ;
			}
			return result ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions+1> :::pushBack(Float const & value) const
		///
		/// \brief	Creates a new vector by pushing the given value at the end of this vector.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \param	value	The added value.
		///
		/// \return	The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions+1> pushBack(Float const & value) const
		{
			return insert(dimensions, value) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions+1> :::pushFront(Float const & value) const
		///
		/// \brief	Creates a new vector by pushing a value at the beginning of this vector.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \param	value	The value to push.
		///
		/// \return The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions+1> pushFront(Float const & value) const
		{
			return insert(0, value) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions-1> :::popBack() const
		///
		/// \brief	Creates a new vector by removing the last coordinate.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \return	The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions-1> popBack() const
		{
			return remove(dimensions-1) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions-1> :::popFront() const
		///
		/// \brief	Creates a new vector by removing the first coordinate.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \return	The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions-1> popFront() const
		{
			return remove(0) ;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \fn	Math::Vector<Float, dimensions> simdMul(Math::Vector<Float, dimensions> const & v) const
		///
		/// \brief	Computes the multiplication coordinate by coordinate of two vectors.
		///
		/// \author	Fabrice Lamarche, university of Rennes 1
		/// \return	The new vector.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Math::Vector<Float, dimensions> simdMul(Math::Vector<Float, dimensions> const & v) const
		{
			Math::Vector<Float, dimensions> result;
			for (int cpt = 0; cpt < dimensions; ++cpt)
			{
				result[cpt] = (*this)[cpt] * v[cpt];
			}
			return result;
		}
	} ;

	template <class Float>
	__device__ __host__ Vector<Float, 3> operator^(const Vector<Float, 3> & v1, const Vector<Float, 3> & v2)
	{
		Vector<Float,3> result ;
		result[0] = v1[1]*v2[2]-v1[2]*v2[1] ;
		result[1] = v1[2]*v2[0]-v1[0]*v2[2] ;
		result[2] = v1[0]*v2[1]-v1[1]*v2[0] ;
		return result ;
	}

	template <class Float>
	__device__ __host__ Vector<Float, 2> makeVector(Float const & c1, Float const & c2)
	{
		Float table[] = {c1, c2} ;
		return Math::Vector<Float, 2>(table) ;
	}

	template <class Float>
	__device__ __host__ Vector<Float, 3> makeVector(Float const & c1, Float const & c2, Float const & c3)
	{
		Float table[] = {c1, c2, c3} ;
		return Math::Vector<Float, 3>(table) ;
	}

	template <class Float>
	Vector<Float, 4> makeVector(Float const & c1, Float const & c2, Float const & c3, Float const & c4)
	{
		Float table[] = {c1, c2, c3, c4} ;
		return Math::Vector<Float, 4>(table) ;
	}

	template <class Float>
	Vector<Float, 5> makeVector(Float const & c1, Float const & c2, Float const & c3, Float const & c4, Float const & c5)
	{
		Float table[] = {c1, c2, c3, c4, c5} ;
		return Math::Vector<Float, 5>(table) ;
	}

	template <class Float, int dimensions>
	inline std::ostream & operator<< (std::ostream & out, Vector<Float, dimensions> const & v)
	{
		for(int cpt=0 ; cpt<dimensions ; cpt++)
		{
			out<<v[cpt]<<" " ;
		}
		return out ;
	}



} 


#endif
