// values for position, color, texcoords, normal for pyradmid and cube

// PYRAMID
// position
float pyramidVertices[] = 
{
	 // front
	 0.0f,  1.0f,  0.0f, // front-top
	-1.0f, -1.0f,  1.0f, // front-left
	 1.0f, -1.0f,  1.0f, // front-right
	 
	 // right
	 0.0f,  1.0f,  0.0f, // right-top
	 1.0f, -1.0f,  1.0f, // right-left
	 1.0f, -1.0f, -1.0f, // right-right

	 // back
	 0.0f,  1.0f,  0.0f, // back-top
	 1.0f, -1.0f, -1.0f, // back-left
	-1.0f, -1.0f, -1.0f, // back-right

	 // left
	 0.0f,  1.0f,  0.0f, // left-top
	-1.0f, -1.0f, -1.0f, // left-left
	-1.0f, -1.0f,  1.0f, // left-right
};

// color
float pyramidColors[] =
{
	// front
	1.0f, 0.0f, 0.0f, // front-top
	0.0f, 1.0f, 0.0f, // front-left
	0.0f, 0.0f, 1.0f, // front-right
	
	// right
	1.0f, 0.0f, 0.0f, // right-top
	0.0f, 0.0f, 1.0f, // right-left
	0.0f, 1.0f, 0.0f, // right-right
	
	// back
	1.0f, 0.0f, 0.0f, // back-top
	0.0f, 1.0f, 0.0f, // back-left
	0.0f, 0.0f, 1.0f, // back-right
	
	// left
	1.0f, 0.0f, 0.0f, // left-top
	0.0f, 0.0f, 1.0f, // left-left
	0.0f, 1.0f, 0.0f, // left-right
};

// texcoords
float pyramidTexcoords[] =
{
	// front
	0.5, 1.0, // front-top
	0.0, 0.0, // front-left
	1.0, 0.0, // front-right

	// right
	0.5, 1.0, // right-top
	1.0, 0.0, // right-left
	0.0, 0.0, // right-right

	// back
	0.5, 1.0, // back-top
	0.0, 0.0, // back-left
	1.0, 0.0, // back-right

	// left
	0.5, 1.0, // left-top
	1.0, 0.0, // left-left
	0.0, 0.0, // left-right
};

// normals
float pyramidNormals[] =
{
	 // front
	 0.000000f, 0.447214f,  0.894427f, // front-top
	 0.000000f, 0.447214f,  0.894427f, // front-left
	 0.000000f, 0.447214f,  0.894427f, // front-right
						    
	 // right			    
	 0.894427f, 0.447214f,  0.000000f, // right-top
	 0.894427f, 0.447214f,  0.000000f, // right-left
	 0.894427f, 0.447214f,  0.000000f, // right-right

	 // back
	 0.000000f, 0.447214f, -0.894427f, // back-top
	 0.000000f, 0.447214f, -0.894427f, // back-left
	 0.000000f, 0.447214f, -0.894427f, // back-right

	 // left
	-0.894427f, 0.447214f,  0.000000f, // left-top
	-0.894427f, 0.447214f,  0.000000f, // left-left
	-0.894427f, 0.447214f,  0.000000f, // left-right
};

float pyramid_PCNT[] =
{
	 // front
	 // position		 // color			// normals						  // texcoords
	 0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.000000f, 0.447214f,  0.894427f, 0.5, 1.0,
	-1.0f, -1.0f,  1.0f, 0.0f, 1.0f, 0.0f,  0.000000f, 0.447214f,  0.894427f, 0.0, 0.0,
	 1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,  0.000000f, 0.447214f,  0.894427f, 1.0, 0.0,
																   
	 // right													   
	 // position		 // color			// normals						  // texcoords
	 0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.894427f, 0.447214f,  0.000000f, 0.5, 1.0,
	 1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,  0.894427f, 0.447214f,  0.000000f, 1.0, 0.0,
	 1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  0.894427f, 0.447214f,  0.000000f, 0.0, 0.0,

	 // back			  
	 // position		 // color			// normals						  // texcoords
	 0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.000000f, 0.447214f, -0.894427f, 0.5, 1.0,
	 1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  0.000000f, 0.447214f, -0.894427f, 0.0, 0.0,
	-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  0.000000f, 0.447214f, -0.894427f, 1.0, 0.0,

	 // left
	 // position		 // color			// normals						  // texcoords
	 0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, -0.894427f, 0.447214f,  0.000000f, 0.5, 1.0,
	-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -0.894427f, 0.447214f,  0.000000f, 1.0, 0.0,
	-1.0f, -1.0f,  1.0f, 0.0f, 1.0f, 0.0f, -0.894427f, 0.447214f,  0.000000f, 0.0, 0.0,
};

// CUBE :

// position
float cubeVertices[] =
{
	// front
	// triangle one
	 1.0f,  1.0f,  1.0f, // top-right of front
	-1.0f,  1.0f,  1.0f, // top-left of front
  	 1.0f, -1.0f,  1.0f, // bottom-right of front
	
	// triangle two
	 1.0f, -1.0f,  1.0f, // bottom-right of front
	-1.0f,  1.0f,  1.0f, // top-left of front
	-1.0f, -1.0f,  1.0f, // bottom-left of front

	// right
	// triangle one
	 1.0f,  1.0f, -1.0f, // top-right of right
	 1.0f,  1.0f,  1.0f, // top-left of right
	 1.0f, -1.0f, -1.0f, // bottom-right of right
	 
	// triangle two
	 1.0f, -1.0f, -1.0f, // bottom-right of right
	 1.0f,  1.0f,  1.0f, // top-left of right
	 1.0f, -1.0f,  1.0f, // bottom-left of right

	// back
	// triangle one
	 1.0f,  1.0f, -1.0f, // top-right of back
	-1.0f,  1.0f, -1.0f, // top-left of back
	 1.0f, -1.0f, -1.0f, // bottom-right of back
	
	// triangle two
	 1.0f, -1.0f, -1.0f, // bottom-right of back
	-1.0f,  1.0f, -1.0f, // top-left of back
	-1.0f, -1.0f, -1.0f, // bottom-left of back

	// left
	// triangle one
	-1.0f,  1.0f,  1.0f, // top-right of left
	-1.0f,  1.0f, -1.0f, // top-left of left
	-1.0f, -1.0f,  1.0f, // bottom-right of left
	
	// triangle two
	-1.0f, -1.0f,  1.0f, // bottom-right of left
	-1.0f,  1.0f, -1.0f, // top-left of left
	-1.0f, -1.0f, -1.0f, // bottom-left of left

	// top
	// triangle one
	 1.0f,  1.0f, -1.0f, // top-right of top
	-1.0f,  1.0f, -1.0f, // top-left of top
	 1.0f,  1.0f,  1.0f, // bottom-right of top

	// triangle two
	 1.0f,  1.0f,  1.0f, // bottom-right of top
	-1.0f,  1.0f, -1.0f, // top-left of top
	-1.0f,  1.0f,  1.0f, // bottom-left of top

	// bottom
	// triangle one
	 1.0f, -1.0f,  1.0f, // top-right of bottom
	-1.0f, -1.0f,  1.0f, // top-left of bottom
	 1.0f, -1.0f, -1.0f, // bottom-right of bottom
	
	// triangle two
	 1.0f, -1.0f, -1.0f, // bottom-right of bottom
	-1.0f, -1.0f,  1.0f, // top-left of bottom
	-1.0f, -1.0f, -1.0f, // bottom-left of bottom
};

float cubeColors[] =
{
	// front
	// triangle one
	1.0f, 0.0f, 0.0f, // top-right of front
	1.0f, 0.0f, 0.0f, // top-left of front
	1.0f, 0.0f, 0.0f, // bottom-right of front

	// triangle two
	1.0f, 0.0f, 0.0f, // bottom-right of front
	1.0f, 0.0f, 0.0f, // top-left of front
	1.0f, 0.0f, 0.0f, // bottom-left of front

	// right
	// triangle one
	0.0f, 0.0f, 1.0f, // top-right of right
	0.0f, 0.0f, 1.0f, // top-left of right
	0.0f, 0.0f, 1.0f, // bottom-right of right
	
	// triangle two
	0.0f, 0.0f, 1.0f, // bottom-right of right
	0.0f, 0.0f, 1.0f, // top-left of right
	0.0f, 0.0f, 1.0f, // bottom-left of right

	// back
	// triangle one
	1.0f, 1.0f, 0.0f, // top-right of back
	1.0f, 1.0f, 0.0f, // top-left of back
	1.0f, 1.0f, 0.0f, // bottom-right of back
	
	// triangle two
	1.0f, 1.0f, 0.0f, // bottom-right of back
	1.0f, 1.0f, 0.0f, // top-left of back
	1.0f, 1.0f, 0.0f, // bottom-left of back

	// left
	// triangle one
	1.0f, 0.0f, 1.0f, // top-right of left
	1.0f, 0.0f, 1.0f, // top-left of left
	1.0f, 0.0f, 1.0f, // bottom-right of left

	// triangle two
	1.0f, 0.0f, 1.0f, // bottom-right of left
	1.0f, 0.0f, 1.0f, // top-left of left
	1.0f, 0.0f, 1.0f, // bottom-left of left

	// top
	// triangle one
	0.0f, 1.0f, 0.0f, // top-right of top
	0.0f, 1.0f, 0.0f, // top-left of top
	0.0f, 1.0f, 0.0f, // bottom-right of top

	// triangle two
	0.0f, 1.0f, 0.0f, // bottom-right of top
	0.0f, 1.0f, 0.0f, // top-left of top
	0.0f, 1.0f, 0.0f, // bottom-left of top

	// bottom
	// triangle one
	1.0f, 0.5f, 0.0f, // top-right of bottom
	1.0f, 0.5f, 0.0f, // top-left of bottom
	1.0f, 0.5f, 0.0f, // bottom-right of bottom

	// triangle two
	1.0f, 0.5f, 0.0f, // bottom-right of bottom
	1.0f, 0.5f, 0.0f, // top-left of bottom
	1.0f, 0.5f, 0.0f, // bottom-left of bottom
};

float cubeTexcoords[] =
{
	// front
	// triangle one
	1.0f, 1.0f, // top-right of front
	0.0f, 1.0f, // top-left of front
	1.0f, 0.0f, // bottom-right of front

	// triangle two
	1.0f, 0.0f, // bottom-right of front
	0.0f, 1.0f, // top-left of front
	0.0f, 0.0f, // bottom-left of front

	// right
	// triangle one
	1.0f, 1.0f, // top-right of right
	0.0f, 1.0f, // top-left of right
	1.0f, 0.0f, // bottom-right of right
	
	// triangle two
	1.0f, 0.0f, // bottom-right of right
	0.0f, 1.0f, // top-left of right
	0.0f, 0.0f, // bottom-left of right

	// back
	// triangle one
	1.0f, 1.0f, // top-right of back
	0.0f, 1.0f, // top-left of back
	1.0f, 0.0f, // bottom-right of back

	// triangle two
	1.0f, 0.0f, // bottom-right of back
	0.0f, 1.0f, // top-left of back
	0.0f, 0.0f, // bottom-left of back

	// left
	// triangle one
	1.0f, 1.0f, // top-right of left
	0.0f, 1.0f, // top-left of left
	1.0f, 0.0f, // bottom-right of left
	
	// triangle two
	1.0f, 0.0f, // bottom-right of left
	0.0f, 1.0f, // top-left of left
	0.0f, 0.0f, // bottom-left of left

	// top
	// triangle one
	1.0f, 1.0f, // top-right of top
	0.0f, 1.0f, // top-left of top
	1.0f, 0.0f, // bottom-right of top

	// triangle two
	1.0f, 0.0f, // bottom-right of top
	0.0f, 1.0f, // top-left of top
	0.0f, 0.0f, // bottom-left of top

	// bottom
	// triangle one
	1.0f, 1.0f, // top-right of bottom
	0.0f, 1.0f, // top-left of bottom
	1.0f, 0.0f, // bottom-right of bottom

	// triangle two
	1.0f, 0.0f, // bottom-right of bottom
	0.0f, 1.0f, // top-left of bottom
	0.0f, 0.0f, // bottom-left of bottom
};

float cubeNormals[] =
{
	 // front surface
	 // triangle one
	 0.0f,  0.0f,  1.0f, // top-right of front
	 0.0f,  0.0f,  1.0f, // top-left of front
	 0.0f,  0.0f,  1.0f, // bottom-right of front
	 
	 // triangle two
	 0.0f,  0.0f,  1.0f, // bottom-right of front
	 0.0f,  0.0f,  1.0f, // top-left of front
	 0.0f,  0.0f,  1.0f, // bottom-left of front

	 // right surface
	 // triangle one
	 1.0f,  0.0f,  0.0f, // top-right of right
	 1.0f,  0.0f,  0.0f, // top-left of right
	 1.0f,  0.0f,  0.0f, // bottom-right of right

	 // triangle two
	 1.0f,  0.0f,  0.0f, // bottom-right of right
	 1.0f,  0.0f,  0.0f, // top-left of right
	 1.0f,  0.0f,  0.0f, // bottom-left of right

	 // back surface
	 // triangle one
	 0.0f,  0.0f, -1.0f, // top-right of back
	 0.0f,  0.0f, -1.0f, // top-left of back
	 0.0f,  0.0f, -1.0f, // bottom-right of back

	 // triangle two
	 0.0f,  0.0f, -1.0f, // bottom-right of back
	 0.0f,  0.0f, -1.0f, // top-left of back
	 0.0f,  0.0f, -1.0f, // bottom-left of back

	 // left surface
	 // triangle one
	-1.0f,  0.0f,  0.0f, // top-right of left
	-1.0f,  0.0f,  0.0f, // top-left of left
	-1.0f,  0.0f,  0.0f, // bottom-right of left

	 // triangle two
	-1.0f,  0.0f,  0.0f, // bottom-right of left
	-1.0f,  0.0f,  0.0f, // top-left of left
	-1.0f,  0.0f,  0.0f, // bottom-left of left

	 // top surface
	 // triangle one
	 0.0f,  1.0f,  0.0f, // top-right of top
	 0.0f,  1.0f,  0.0f, // top-left of top
	 0.0f,  1.0f,  0.0f, // bottom-right of top

	 // triangle two
	 0.0f,  1.0f,  0.0f, // bottom-right of top
	 0.0f,  1.0f,  0.0f, // top-left of top
	 0.0f,  1.0f,  0.0f, // bottom-left of top

	 // bottom surface
	 // triangle one
	 0.0f, -1.0f,  0.0f, // top-right of bottom
	 0.0f, -1.0f,  0.0f, // top-left of bottom
	 0.0f, -1.0f,  0.0f, // bottom-right of bottom
	 
	 // triangle two
	 0.0f, -1.0f,  0.0f, // bottom-right of bottom
	 0.0f, -1.0f,  0.0f, // top-left of bottom
	 0.0f, -1.0f,  0.0f, // bottom-left of bottom
};

float cube_PCNT[] =
{
	// front
	// position				// color			 // normals				// texcoords
	 1.0f,  1.0f,  1.0f,	1.0f, 0.0f, 0.0f,	 0.0f,  0.0f,  1.0f,	1.0f, 1.0f,	 // top-right
	-1.0f,  1.0f,  1.0f,	1.0f, 0.0f, 0.0f,	 0.0f,  0.0f,  1.0f,	0.0f, 1.0f,	 // top-left
	 1.0f, -1.0f,  1.0f,	1.0f, 0.0f, 0.0f,	 0.0f,  0.0f,  1.0f,	1.0f, 0.0f,	 // bottom-right

	 1.0f, -1.0f,  1.0f,	1.0f, 0.0f, 0.0f,	 0.0f,  0.0f,  1.0f,	1.0f, 0.0f,	 // bottom-right
	-1.0f,  1.0f,  1.0f,	1.0f, 0.0f, 0.0f,	 0.0f,  0.0f,  1.0f,	0.0f, 1.0f,	 // top-left
	-1.0f, -1.0f,  1.0f,	1.0f, 0.0f, 0.0f,	 0.0f,  0.0f,  1.0f,	0.0f, 0.0f,  // bottom-left
						 
	// right			 
	// position				// color			 // normals				// texcoords
	 1.0f,  1.0f, -1.0f,	0.0f, 0.0f, 1.0f,	 1.0f,  0.0f,  0.0f,	1.0f, 1.0f,  // top-right
	 1.0f,  1.0f,  1.0f,	0.0f, 0.0f, 1.0f,	 1.0f,  0.0f,  0.0f,	0.0f, 1.0f,  // top-left
	 1.0f, -1.0f, -1.0f,	0.0f, 0.0f, 1.0f,	 1.0f,  0.0f,  0.0f,	1.0f, 0.0f,  // bottom-right

	 1.0f, -1.0f, -1.0f,	0.0f, 0.0f, 1.0f,	 1.0f,  0.0f,  0.0f,	1.0f, 0.0f,  // bottom-right
	 1.0f,  1.0f,  1.0f,	0.0f, 0.0f, 1.0f,	 1.0f,  0.0f,  0.0f,	0.0f, 1.0f,  // top-left
	 1.0f, -1.0f,  1.0f,	0.0f, 0.0f, 1.0f,	 1.0f,  0.0f,  0.0f,	0.0f, 0.0f,  // bottom-left
						 
	// back				 
	// position				// color			 // normals				// texcoords
	 1.0f,  1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	 0.0f,  0.0f, -1.0f,	1.0f, 1.0f,  // top-right
	-1.0f,  1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	 0.0f,  0.0f, -1.0f,	0.0f, 1.0f,  // top-left
     1.0f, -1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	 0.0f,  0.0f, -1.0f,	1.0f, 0.0f,  // bottom-right

     1.0f, -1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	 0.0f,  0.0f, -1.0f,	1.0f, 0.0f,  // bottom-right
	-1.0f,  1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	 0.0f,  0.0f, -1.0f,	0.0f, 1.0f,  // top-left
	-1.0f, -1.0f, -1.0f,	1.0f, 1.0f, 0.0f,	 0.0f,  0.0f, -1.0f,	0.0f, 0.0f,  // bottom-left
						 
	// left				 
	// position				// color			 // normals				// texcoords
	-1.0f,  1.0f,  1.0f,	1.0f, 0.0f, 1.0f,	-1.0f,  0.0f,  0.0f,	1.0f, 1.0f,  // top-right
	-1.0f,  1.0f, -1.0f,	1.0f, 0.0f, 1.0f,	-1.0f,  0.0f,  0.0f,	0.0f, 1.0f,  // top-left
	-1.0f, -1.0f,  1.0f,	1.0f, 0.0f, 1.0f,	-1.0f,  0.0f,  0.0f,	1.0f, 0.0f,  // bottom-right

	-1.0f, -1.0f,  1.0f,	1.0f, 0.0f, 1.0f,	-1.0f,  0.0f,  0.0f,	1.0f, 0.0f,  // bottom-right
	-1.0f,  1.0f, -1.0f,	1.0f, 0.0f, 1.0f,	-1.0f,  0.0f,  0.0f,	0.0f, 1.0f,  // top-left
	-1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 1.0f,	-1.0f,  0.0f,  0.0f,	0.0f, 0.0f,  // bottom-left
						 
	// top				 
	// position				// color			 // normals				// texcoords
	 1.0f,  1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	 0.0f,  1.0f,  0.0f,	1.0f, 1.0f,  // top-right
	-1.0f,  1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	 0.0f,  1.0f,  0.0f,	0.0f, 1.0f,  // top-left
     1.0f,  1.0f,  1.0f,	0.0f, 1.0f, 0.0f,	 0.0f,  1.0f,  0.0f,	1.0f, 0.0f,  // bottom-right

     1.0f,  1.0f,  1.0f,	0.0f, 1.0f, 0.0f,	 0.0f,  1.0f,  0.0f,	1.0f, 0.0f,  // bottom-right
	-1.0f,  1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	 0.0f,  1.0f,  0.0f,	0.0f, 1.0f,  // top-left
	-1.0f,  1.0f,  1.0f,	0.0f, 1.0f, 0.0f,	 0.0f,  1.0f,  0.0f,	0.0f, 0.0f,  // bottom-left
						 
	// bottom			 
	// position				// color			 // normals				// texcoords
	 1.0f, -1.0f,  1.0f,	1.0f, 0.5f, 0.0f,	 0.0f, -1.0f,  0.0f,	1.0f, 1.0f, // top-right
	-1.0f, -1.0f,  1.0f,	1.0f, 0.5f, 0.0f,	 0.0f, -1.0f,  0.0f,	0.0f, 1.0f, // top-left
	 1.0f, -1.0f, -1.0f,	1.0f, 0.5f, 0.0f,	 0.0f, -1.0f,  0.0f,	1.0f, 0.0f, // bottom-right

	 1.0f, -1.0f, -1.0f,	1.0f, 0.5f, 0.0f,	 0.0f, -1.0f,  0.0f,	1.0f, 0.0f, // bottom-right
	-1.0f, -1.0f,  1.0f,	1.0f, 0.5f, 0.0f,	 0.0f, -1.0f,  0.0f,	0.0f, 1.0f, // top-left
	-1.0f, -1.0f, -1.0f,	1.0f, 0.5f, 0.0f,	 0.0f, -1.0f,  0.0f,	0.0f, 0.0f, // bottom-left
};
