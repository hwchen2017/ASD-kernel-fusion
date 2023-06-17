
typedef struct v3d{
	double x, y, z; 

}v3d;

v3d make_v3d(const double _x, const double _y, const double _z)
{
	v3d tmp; 
	tmp.x = _x; 
	tmp.y = _y; 
	tmp.z = _z; 
	return tmp; 
}

void v3d_minus(v3d* a, v3d* b)
{
	a->x -= b->x; 
	a->y -= b->y; 
	a->z -= b->z; 
}

void v3d_add(v3d* a, v3d* b)
{
	a->x += b->x; 
	a->y += b->y; 
	a->z += b->z; 
}

void v3d_multiply(v3d* a, double coeff)
{
	a->x *= coeff; 
	a->y *= coeff; 
	a->z *= coeff; 
}

double v3d_dot(v3d* a, v3d* b)
{
	double res = 0.0; 

	res += a->x * b->x; 
	res += a->y * b->y; 
	res += a->z * b->z;

	return res;  
}

void v3d_cross(v3d* res, v3d* a, v3d* b)
{ 
	res->x = a->y * b->z - a->z * b->y; 
    res->y = a->z * b->x - a->x * b->z; 
    res->z = a->x * b->y - a->y * b->x; 
}
