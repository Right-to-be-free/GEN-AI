square = lambda x: x**2
print(square(5))


counter = 0

def increment():
    global counter
    counter += 1
increment()    
print(counter) 

#discount = 10%
def cal_discount(price, discount):
    return price - price * discount / 100
print(cal_discount(100, 10))

#oops concept in python simplify the code and make it more readable.
#artibutes can have diffrent level of axcess control.
#class is a blueprint for creating objects.
#objects are instances of class.
# public attributes can be accessed from outside the class.
#private attributes can only be accessed from within the class.
#protected attributes can be accessed from within the class and its subclasses.
#methods are functions defined inside the class.
#constructor is a special method that gets called when an object is created.
#inheritance is a way to create a new class using an existing class.
#polymorphism allows methods to do different things based on the object.
#encapsulation is a way to restrict access to some parts of the object.
# class variables are shared among all instances of a class.
# instance variables are unique to each instance of a class.

class Person:
    def __init__(self, name, email):
        self.name = name
        self._email = email
        self.__password = "secret"


# constructor is a special method that gets called when an object is created.
# destructor is a special method that gets called when an object is deleted.
    