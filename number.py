import random
def gen_problem():
    kind = random.choice(['add','sub','mul','div'])
    while True:
        a,b = [random.randrange(-99,99) for _ in range(2)]
        match kind:
            case 'add':
                if a+b in range(10):
                    return f"{a} + {b} = {a+b}"
            case 'sub':
                if a-b in range(10):
                    return f"{a} - {b} = {a-b}"
            case 'div':
                if a//b in range(10):
                    return f"{a} / {b} = {a//b}"
            case 'mul':
                if a*b in range(10):
                    return f"{a} * {b} = {a*b}"
print([gen_problem() for _ in range(10)])
