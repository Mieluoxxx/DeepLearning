import time


def display_time(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print("Total time: {:.4} s".format(end-start))
        return result
    return wrapper


def is_prime(num):
    if num < 2:
        return False
    elif num == 2:
        return True
    else:
        for i in range(2, num):
            if num % i == 0:
                return False
        return True


@display_time
def count_prime_nums(maxnum):
    count = 0
    for i in range(2, maxnum):
        if is_prime(i):
            count += 1
    return count


count = count_prime_nums(5000)
print(count)