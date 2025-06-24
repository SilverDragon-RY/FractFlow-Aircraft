def test():
    yield 1
    yield 1,2

print(test())