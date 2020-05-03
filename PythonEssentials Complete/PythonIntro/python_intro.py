# python_intro.py
"""Python Essentials: Introduction to Python.
Marcelo Leszynski
<Class>
04/19/20
"""
def alt_harmonic(n):
    iterator = [(-1)**(c+1) / c for c in range(1, n+1)]
    return sum(iterator)

def palindrome():
    highest_number = 0
    for i in range(100, 1000):
        #iterate through the first term
        for j in range(i+1, 1000):
            #iterate through the second term
            temp_number = i * j
            if str(temp_number) == str(temp_number)[::-1] and temp_number >= highest_number:
                highest_number = temp_number
    return highest_number

def pig_latin(text):
    vowel_list = ['a', 'e', 'i', 'o', 'u', 'y']
    if text[0] in vowel_list:
        return text + "hay"
    else:
        first_letter = text[0]
        return text[1:] + first_letter + "ay"

def list_ops():
    my_list = ["bear", "ant", "cat", "dog"]
    my_list.append("eagle")
    my_list[2]="fox"
    my_list.remove(my_list[1])
    my_list.sort(reverse=True)
    my_list[my_list.index("eagle")]="hawk"
    my_list[-1] += "hunter"
    print(my_list)

def backward(text):
    return text[::-1]

def first_half(text):
    middle = len(text) // 2
    print(middle)
    return text[:middle]

def isolate(a, b, c, d, e):
    print(a, ' '*3, b, ' '*3, c, d, e)

def sphere_volume(r):
    """
    Return the volume of a sphere with radius r using 
    3.14159 as an approximation for pi.
    """
    return (4 / 3) * 3.14159 * (r**3)

if __name__ == "__main__":
    print("Hello, world!")
    print(sphere_volume(27))
    print(isolate(1,2,3,4,5))
    print(first_half("This is some sample text."))
    print(backward("This is some sample text."))
    list_ops()
    print(pig_latin("pizza"))
    print(pig_latin("okay"))
    print(palindrome())
    print(alt_harmonic(500000))