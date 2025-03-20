import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

class vm: 

  DEBUG = True

#
  def add(self, right):
    result = self + right;
    vm.debugprint(self, right, 'SUM', result)
    return result
  
#    
  def sub(self, right):
    result = self - right;
    vm.debugprint(self, right, 'DIFFERENCE', result)
    return result

#
  def multiply(self, right):
    result = np.matmul(self, right)
    vm.debugprint(self, right, 'MULTIPLY', result)
    return result

#
  def scalermult(self, scaler):
    result = self * scaler
    if vm.DEBUG:
      print(self)
      print(f'scaler-> {scaler}')
      print('SCALE')
      print(f'{result}\n')
    return result

# 
  def divide(self, right):
    result = self / right;
    vm.debugprint(self, right, 'QUOTIENT', result)
    return result

#     
  def transpose(self):
    result = self.transpose()
    if vm.DEBUG:
      print(self)
      print('TRANSPOSE')
      print(f'{result}\n')
    return result
  
#
  def cross(self, right):
    result = np.cross(self, right)
    vm.debugprint(self, right, 'CROSS PRODUCT', result)
    return result

#
  def dot(self, right):
    result = np.dot(self, right)
    vm.debugprint(self, right, 'DOT PRODUCT', result)
    return result
  

#


  def rotation2d(self, theta):
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    result = vm.multiply(self, rotationMatrix)
    if vm.DEBUG:
      print(self)
      print(f'theta-> {theta}')
      print('ROTATE')
      print(f'{result}\n')  
    return result
#  
  def rotation3d(self, axis, angle):
    if axis == 'x':
      phi = angle
      rotationMatrix = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    elif axis == 'y':
      theta = angle
      rotationMatrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
      psi = angle
      rotationMatrix = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    else:
      print('ERROR in rotation matrix dependant axis')
      exit(-1)
    vm.DEBUG = False
    result = vm.multiply(self, rotationMatrix)
    vm.DEBUG = True
    angle
    vm.debugprint(self,rotationMatrix,'ROTATION ' + axis + ' ' + str(angle) + ' 3D', result)
    return result


#
  def debugprint(self, right, label, solution):
    if vm.DEBUG:
      print(self)
      if self.ndim > 1:
        print(' ')
      print(right)
      print(label)
      print(f'{solution}\n')
      
      
if __name__ == '__main__':
  uMatrix33 = np.random.randint(-20, 20, size=(3, 3))
  vMatrix33 = np.random.randint(-20, 20, size=(3, 3))
  wMatrix33 = np.random.randint(-20, 20, size=(3, 3))
  xMatrix33 = np.random.randint(-20, 20, size=(3, 3)) 
  yMatrix33 = np.random.randint(-20, 20, size=(3, 3)) 
  zMatrix33 = np.random.randint(-20, 20, size=(3, 3))

  uMatrix22 = np.random.randint(-20, 20, size=(2, 2))
  vMatrix22 = np.random.randint(-20, 20, size=(2, 2))
  wMatrix22 = np.random.randint(-20, 20, size=(2, 2))
  xMatrix22 = np.random.randint(-20, 20, size=(2, 2)) 
  yMatrix22 = np.random.randint(-20, 20, size=(2, 2))
  zMatrix22 = np.random.randint(-20, 20, size=(2, 2))

  uVector = np.random.randint(-20, 20, size=(3))
  vVector = np.random.randint(-20, 20, size=(3))
  wVector = np.random.randint(-20, 20, size=(3))
  xVector = np.random.randint(-20, 20, size=(3))
  yVector = np.random.randint(-20, 20, size=(3))
  zVector = np.random.randint(-20, 20, size=(3))
 

  vm.add(uMatrix33, vMatrix33)
  vm.sub(vMatrix33, wMatrix33)
  vm.multiply(wMatrix33, xMatrix33)
  vm.scalermult(zMatrix33, 3)
  vm.divide(xMatrix33, yMatrix33)

  vm.transpose(zMatrix33)

  crossVector = vm.cross(uVector, vVector)
  print('If the cross product is linearly independent,\n the dot product should equal 0')
  vm.DEBUG = False
  result = vm.dot(vVector, crossVector)
  if result== 0:
    print('PASSED, vector is linearly independent')
  else:
    print('FAILED, vector is NOT linearly dependent')
  vm.DEBUG = True
  vm.dot(vVector, crossVector)

  vm.dot(vVector, wVector)

#
# Perform a sequence of matrix rotations around a selected axis. Initially 2D matricies followed
# by 3D matricies

  print('The following section is to test Matrix Rotation methods for 2x2 and 3x3 matricies.\n \
  2x2 matricies rotation around the Z axis "Yaw". 3x3 matricies will rotate around X "Roll", Y "Pitch", \
  and Z "Yaw"\n axies individually. Axis of rotation is user specified.\n')

# 2D rotations
  theta = np.radians(180)
  rotatedMatrix = vm.rotation2d(xMatrix22, theta)

# 3D rotations
  angle = np.radians(90)
  rootatedMatrix = vm.rotation3d(zMatrix33, 'x', angle)
  angle = np.radians(180)
  rootatedMatrix = vm.rotation3d(yMatrix33, 'y', angle)
  angle = np.radians(270)
  rotatedMatrix = vm.rotation3d(xMatrix33, 'z', angle)

#
# The following section is for testing inverse matrix generation
# the identity matrix should be the product of matrix multiplication

  print('Want to test if matricies are inverse of each other\n The product should generate an identitiy matrix')
  aMatrix = np.array([[1, 2, -1], [-2, 0, 1], [1, -1, 0]])
  bMatrix = np.array([[1, 1, 2], [1, 1, 1], [2, 3, 4]])

  cMatrix = np.array([[1, 0, 2],[2, -1,3],[4, 1, 8]])
  dMatrix = np.array([[-11, 2, 2], [-4, 0, 1], [6, -1, -1]])

# Generate a matrix of random integers using the numpy random integer method.
# Compute the inverse matrix using "inv()" from the numpy, 
# linear algebra package library
  eMatrix = np.random.randint(-20, 20, size=(3, 3))
  laInvMatrix = np.linalg.inv(eMatrix)

  vm.multiply(aMatrix, bMatrix)
  vm.multiply(cMatrix, dMatrix)
  vm.multiply(eMatrix, laInvMatrix)

#
# Moving on to actually graphically plotting vectors using matplotlib with pyplot
#
  fig = plt.figure()
  axis = fig.add_subplot(111, projection='3d')
  axis.set_xlim([-20, 20])
  axis.set_ylim([-20, 20])
  axis.set_zlim([-20, 20])
  axis.set_xlabel('X')
  axis.set_ylabel('Y')
  axis.set_zlabel('Z')
  axis.set_title('3D VECTOR PLOTS')

  startAt = [0, 0, 0]
  axis.quiver3D(startAt[0], startAt[1], startAt[2], xVector[0], xVector[1], xVector[2], arrow_length_ratio=0.4).set_color('black')
  axis.quiver3D(startAt[0], startAt[1], startAt[2], yVector[0], yVector[1], yVector[2], arrow_length_ratio=0.4).set_color('green')
  orthogonalVector = vm.cross(xVector, yVector)
  axis.quiver3D(startAt[0], startAt[1], startAt[2], orthogonalVector[0], orthogonalVector[1], orthogonalVector[2], arrow_length_ratio=0.4).set_color('red')
  dotProduct = vm.dot(orthogonalVector, xVector)
  print(f'dot product-> {dotProduct}')
  plt.show()

