def demonstrate_exceptions():
    my_list = [1, 2, 3]

    try:
        print(my_list[5])
    except IndexError:
        print("IndexError occurred! List index out of range.")

    my_dict = {'a': 1, 'b': 2}

    try:
        print(my_dict['c'])
    except KeyError:
        print("KeyError occurred! Key not found in the dictionary.")

    my_string = "Hello"
    my_number = 5

    try:
        print(my_string + my_number)
    except TypeError:
        print("TypeError occurred! Unsupported operand types.")

demonstrate_exceptions()
