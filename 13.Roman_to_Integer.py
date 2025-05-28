def romanToInt(s) -> int:
        #save a chart to convert letters to numbers
    letters = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000,
            
        }
        
    total = 0
        
    values = []

    for characters in s:
        values.append(letters[characters])
          
    for i in range(len(values) -1):
            
        # if i+1 < len(values):
        if values[i] < values[i+1]:
            total -= values[i]
        else:
            total += values[i]
    
    last_value = total + values[-1]

       
    return last_value
                    

# print(romanToInt('XIX'))
print(romanToInt('MCMXCIV'))
