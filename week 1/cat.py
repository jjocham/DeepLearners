
# Cat CLASS YAY
# define a "cat class" with a constructor that lets you name a "cat" 
# and a method that lets the cats greet each other

class cats:

    def __init__(self, name):
        self.name = name

    
    def greet(self, other):
        print("Hello furry friend, I'm " + self.name + ", your name is " + other.name + ", right?")

