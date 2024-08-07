import hashlib  
import numpy as np

def repeatable_random(seed, num_numbers):
    numbers = []
    temp = []
    hash = str(seed).encode()  
    while len(numbers) < num_numbers:
        hash = hashlib.md5(hash).digest()
        for c in hash:
            temp.append(c)
            if(len(temp) == 4):
                num = int(''.join([str(num) for num in temp]))
                temp = []
                numbers.append(num)
            if len(numbers) >= num_numbers:
                break 
    return numbers

# num_sims = np.ceil(2_500_342/1_000_000)
# print(num_sims)
print(repeatable_random(1234,10))
