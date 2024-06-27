from stl import mesh
import math
import numpy 

# # Create 3 faces of a cube
data = numpy.zeros(6, dtype=mesh.Mesh.dtype)


# Top of the cube
data['vectors'][0] = numpy.array([[0, 1, 1],
                                  [1, 0, 1],
                                  [0, 0, 1]])
data['vectors'][1] = numpy.array([[1, 0, 1],
                                  [0, 1, 1],
                                  [1, 1, 1]])
# Front face
data['vectors'][2] = numpy.array([[1, 0, 0],
                                  [1, 0, 1],
                                  [1, 1, 0]])
data['vectors'][3] = numpy.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 0]])
# Left face
data['vectors'][4] = numpy.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 0, 1]])
data['vectors'][5] = numpy.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [1, 0, 1]])
# Generate 4 different meshes so we can rotate them later
meshes = mesh.Mesh(data.copy()) 
print(type(meshes))
print('the shape of meshes is', numpy.shape(meshes))
print(meshes.vectors)
meshes[2].y +=2
print(meshes[2])
# ####################
# data_try = np.array([[[1,2,3], [4,5,6], [1,2,3]],[[7,8,9], [10,11,12], [13,14,15]]])
# print(np.shape(data_try))
# for i in range(len(data_try)):
# 	print('the vector in the mesh is',data_try[i])
# 	for j in range(len(data_try[i])):
# 		print('the number of vector in the mesh is',data_try[i][j])
# 		print('the y value for each vector is',data_try[i][j][1])
# print(data_try[:][:][1])

