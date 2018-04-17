import numpy as np
#print( 10000 // 1000)

input=[]
input.append([1,2,3]) #id
input.append([4,5,6])
print(np.array(input[0]))
inputs = np.transpose(np.vstack([np.array(input[i]) for i in range(len(input))]))
print(inputs)
print(len(inputs[0]))

ids = np.random.randint(0, 10000, (100,))
print("ids:",ids)