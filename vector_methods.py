import numpy as np
import os

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
    pass

#  
  def rotation3d(self, axis, theta):
    pass

#
  def debugprint(self, right, label, solution):
    if vm.DEBUG:
      print(self)
      if self.ndim > 1:
        print(' ')
      print(right)
      print(label)
      print(solution)
      print('\n')   

if __name__ == '__main__':
  uMatrix = np.random.randint(-20, 20, size=(3, 3))
  vMatrix = np.random.randint(-20, 20, size=(3, 3))
  wMatrix = np.random.randint(-20, 20, size=(3, 3))
  xMatrix = np.random.randint(-20, 20, size=(3, 3)) 
  yMatrix = np.random.randint(-20, 20, size=(3, 3))
  zMatrix = np.random.randint(-20, 20, size=(3, 3))

  uVector = np.random.randint(-20, 20, size=(3))
  vVector = np.random.randint(-20, 20, size=(3))
  wVector = np.random.randint(-20, 20, size=(3))
  xVector = np.random.randint(-20, 20, size=(3))
  yVector = np.random.randint(-20, 20, size=(3))
  zVector = np.random.randint(-20, 20, size=(3))
 

  vm.add(uMatrix, vMatrix)
  vm.sub(vMatrix, wMatrix)
  vm.multiply(wMatrix, xMatrix)
  vm.scalermult(zMatrix, 3)
  vm.divide(xMatrix, yMatrix)

  vm.transpose(zMatrix)

  crossVector = vm.cross(uVector, vVector)
  print('If the cross product is linearly independent,\n the dot product should equal 0')
  vm.DEBUG = False
  result = vm.dot(vVector, crossVector)
  if result== 0:
    print('PASSED, vector is linearly independent')
  else:
    print('FAILED, vector is linearly dependent')
  vm.DEBUG = True
  vm.dot(vVector, crossVector)

  vm.dot(vVector, wVector)

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
