

# Implement a generator function which returns a meow the first time
# you call it, and then twice the number of meows on each consecutive call

def meow_generator():
    m_count = 1
    while True:
        yield "meow " * m_count
        m_count *= 2

gen = meow_generator()
for i in range(4):
    print(next(gen))