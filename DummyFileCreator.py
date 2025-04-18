
size = int(input("how many KB to write:\n"))
string = " " * 1024  # 1 KB
try:
    for i in range(size):
        with open("dummy", "a") as file:
            file.write(string)
            file.flush()
        print(f"\r{round((i+1) / size * 100, 0)}%", end="")
except Exception as error:
    print(f"{error}")
    input("Press any key to continue")
