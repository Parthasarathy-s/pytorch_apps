class Value:
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data)
    
    def __mul__(self, other):
        return Value(self.data * other.data)

a = Value(2)
b = Value(-3)
c = Value (10)

print (a * b + c)
print((a.__mul__(b)).__add__(c))
