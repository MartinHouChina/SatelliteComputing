class A:
    def __init__(self, value):
        self.value = value


if __name__ == '__main__':
    a = A(2)
    b = [a.value]
    b[0] -= 2
    print(a.value)
